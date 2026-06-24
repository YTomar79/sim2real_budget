"""Train + evaluate one (n, w, seed) run of the budget-allocation experiment.

Pipeline for a single run:
  1. System identification: spend `n` real rollouts to estimate theta_hat.
  2. Build a domain-randomized training env around theta_hat with width `w`.
  3. Train the learner (SAC by default; PPO optional) in that simulator.
  4. Zero-shot evaluate on the real system (theta_star).
  5. Write result.json (+ model.zip) into the run's output directory.

Invoke either with an explicit config or a flat array index:
  python -m src.train --task_id 17 --output_root /path/to/results/sweep
  python -m src.train --n 10 --w 0.25 --seed 0 --output_root ./results/sweep
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np

from . import config
from .param_pendulum import ParamPendulumEnv
from .dr_wrapper import DomainRandomizationWrapper
from .system_id import identify
from .evaluate import evaluate_on_real


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


def build_train_env(theta_hat, width, seed, dr_mode="relative", max_episode_steps=200):
    base = ParamPendulumEnv(
        mass=theta_hat[0], length=theta_hat[1], max_episode_steps=max_episode_steps
    )
    return DomainRandomizationWrapper(
        base, theta_hat=theta_hat, width=width, seed=seed, mode=dr_mode
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single budget-allocation run.")
    # Either provide --task_id, or all of --n/--w/--seed.
    p.add_argument("--task_id", type=int, default=None,
                   help="flat array index; decoded via config.decode_task_id.")
    p.add_argument("--n", type=int, default=None, help="real rollouts for system ID")
    p.add_argument("--w", type=float, default=None, help="domain randomization width")
    p.add_argument("--seed", type=int, default=None)

    p.add_argument("--output_root", type=str, required=True,
                   help="root dir; a per-run subdir is created under it")

    # HParam overrides
    p.add_argument("--algo", type=str, default=config.HParams.algo,
                   choices=["sac", "ppo"])
    p.add_argument("--total_timesteps", type=int, default=config.HParams.total_timesteps)
    p.add_argument("--n_eval_episodes", type=int, default=config.HParams.n_eval_episodes)
    p.add_argument("--id_rollout_len", type=int, default=config.HParams.id_rollout_len)
    p.add_argument("--id_mode", type=str, default=config.HParams.id_mode,
                   choices=["grid", "ideal"])
    p.add_argument("--id_grid_res", type=int, default=config.HParams.id_grid_res)
    p.add_argument("--id_obs_noise", type=float, default=config.HParams.id_obs_noise)
    p.add_argument("--theta_star", type=str, default=None,
                   help='override the hidden "real" params, format "mass,length" '
                        '(default: config.THETA_STAR)')
    p.add_argument("--dr_mode", type=str, default="relative",
                   choices=["relative", "prior"],
                   help="relative: randomize within +/-w of the estimate; "
                        "prior: ignore the estimate and sample uniformly over "
                        "PARAM_BOUNDS (wide prior that brackets the truth)")
    p.add_argument("--learning_rate", type=float, default=config.HParams.learning_rate)
    p.add_argument("--n_steps", type=int, default=config.HParams.n_steps)
    p.add_argument("--batch_size", type=int, default=config.HParams.batch_size)
    p.add_argument("--gamma", type=float, default=config.HParams.gamma)
    p.add_argument("--ent_coef", type=float, default=config.HParams.ent_coef)
    p.add_argument("--torch_threads", type=int, default=None,
                   help="cap torch intra-op threads (default: leave as-is)")
    return p.parse_args()


def resolve_run(args) -> tuple[int, float, int]:
    if args.task_id is not None:
        return config.decode_task_id(args.task_id)
    if None in (args.n, args.w, args.seed):
        raise SystemExit("Provide either --task_id or all of --n/--w/--seed.")
    return args.n, args.w, args.seed


def main() -> None:
    args = parse_args()
    n, w, seed = resolve_run(args)

    # Optional override of the hidden "real" parameters. make_real_env, identify,
    # and evaluate all read config.THETA_STAR at call time, so setting the module
    # global here is sufficient.
    if args.theta_star is not None:
        parts = [float(x) for x in args.theta_star.split(",")]
        if len(parts) != 2:
            raise SystemExit('--theta_star must be "mass,length", e.g. "2.5,2.0"')
        config.THETA_STAR = (parts[0], parts[1])

    if args.torch_threads is not None:
        try:
            import torch

            torch.set_num_threads(int(args.torch_threads))
        except Exception:
            pass

    hp = config.HParams(
        algo=args.algo,
        total_timesteps=args.total_timesteps,
        n_eval_episodes=args.n_eval_episodes,
        id_rollout_len=args.id_rollout_len,
        id_mode=args.id_mode,
        id_grid_res=args.id_grid_res,
        id_obs_noise=args.id_obs_noise,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
    )

    out_dir = Path(args.output_root) / config.run_name(n, w, seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Mark in-progress so a resubmission can skip completed runs.
    result_path = out_dir / "result.json"
    if result_path.exists():
        try:
            prev = json.loads(result_path.read_text())
            if prev.get("status") == "ok":
                print(f"[skip] {out_dir} already complete.")
                return
        except Exception:
            pass

    t0 = time.time()
    set_global_seeds(seed)

    # 1) System identification
    theta_hat, id_error = identify(n, hp, seed)
    print(f"[id] n={n} mode={hp.id_mode} theta_hat={theta_hat} "
          f"theta_star={config.THETA_STAR} id_error={id_error:.4f}")

    # 2) + 3) Train the learner in the domain-randomized simulator.
    # SAC is the default: off-policy and sample-efficient, it reliably solves
    # Pendulum in ~30-50k steps, whereas vanilla PPO needs several hundred
    # thousand and often stalls near random return (~ -1100).
    train_env = build_train_env(theta_hat, w, seed, dr_mode=args.dr_mode)
    train_env.reset(seed=seed)

    if hp.algo == "sac":
        from stable_baselines3 import SAC  # imported late so --help is fast

        model = SAC(
            "MlpPolicy",
            train_env,
            learning_rate=hp.learning_rate,
            buffer_size=hp.buffer_size,
            learning_starts=hp.learning_starts,
            batch_size=hp.batch_size,
            tau=hp.tau,
            gamma=hp.gamma,
            train_freq=hp.train_freq,
            gradient_steps=hp.gradient_steps,
            seed=seed,
            verbose=0,
        )
    elif hp.algo == "ppo":
        from stable_baselines3 import PPO

        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=hp.learning_rate,
            n_steps=hp.n_steps,
            batch_size=hp.batch_size,
            gamma=hp.gamma,
            ent_coef=hp.ent_coef,
            seed=seed,
            verbose=0,
        )
    else:
        raise ValueError(f"unknown algo '{hp.algo}'")

    model.learn(total_timesteps=hp.total_timesteps, progress_bar=False)

    # 4) Zero-shot evaluation on the real system
    eval_mean, eval_std, eval_returns = evaluate_on_real(
        model, n_eval_episodes=hp.n_eval_episodes, seed=seed
    )
    print(f"[eval] n={n} w={w} seed={seed} "
          f"return={eval_mean:.2f} +/- {eval_std:.2f}")

    # 5) Persist
    model.save(str(out_dir / "model.zip"))
    result = {
        "status": "ok",
        "n": n,
        "w": w,
        "seed": seed,
        "dr_mode": args.dr_mode,
        "theta_star": list(config.THETA_STAR),
        "theta_hat": list(theta_hat),
        "id_error": id_error,
        "eval_return_mean": eval_mean,
        "eval_return_std": eval_std,
        "eval_returns": eval_returns,
        "hparams": hp.to_dict(),
        "wall_time_s": time.time() - t0,
    }
    result_path.write_text(json.dumps(result, indent=2))
    print(f"[done] wrote {result_path} in {result['wall_time_s']:.1f}s")


if __name__ == "__main__":
    main()
