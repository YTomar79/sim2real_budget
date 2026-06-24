"""Single source of truth for the sweep grid and experiment constants.

The whole pipeline is organized around one question:
    Given a fixed real-world interaction budget, how should it be split between
    (a) system identification (improving simulator FIDELITY) and
    (b) reliance on domain-randomization BREADTH (free in sim, costs optimality)?

A "run" is one (n, w, seed) triple:
    n    = number of real-world rollouts spent on system identification
    w    = domain-randomization width (relative, fraction of the point estimate)
    seed = RNG seed

A parallel launcher runs one task per run; `decode_task_id` maps a flat array
index to a concrete (n, w, seed). Keep this file as the ONLY place the grid is
defined so the launcher and the Python code never disagree.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Tuple

# --------------------------------------------------------------------------- #
# Sweep grid
# --------------------------------------------------------------------------- #
# Real-world rollouts allocated to system identification (the scarce budget).
N_GRID = [0, 5, 10, 25, 50]
# Domain-randomization width (relative to the point estimate of each parameter).
# v2: dropped the degenerate w=1.0 (drove params to the 0.3 clip floor -> pathological
# dynamics); finer steps capped at 0.5 so the breadth axis stays physically meaningful.
W_GRID = [0.0, 0.1, 0.2, 0.3, 0.5]
# Seeds: 10 for statistical power on the noisy low-n rows (the cells where an
# interior optimum, if it exists, must show up). 5 x 5 x 10 = 250 jobs.
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# --------------------------------------------------------------------------- #
# "Real" system definition (held out; never revealed to training)
# --------------------------------------------------------------------------- #
# Pendulum physical parameters (mass, length). The "real" robot differs from the
# nominal prior the engineer would assume before any identification.
# v2: widened the gap from the prior so the mismatch actually degrades zero-shot
# transfer at w=0 (with theta*=(1.5,1.2) Pendulum was too forgiving -> no headroom
# for breadth to help). Larger mass also lowers control authority (~1/(m l^2)),
# making transfer more sensitive to the gap.
THETA_STAR: Tuple[float, float] = (2.0, 1.5)   # (mass*, length*)
PRIOR_THETA: Tuple[float, float] = (1.0, 1.0)  # nominal guess used when n == 0

# Physically plausible bounds for both system ID search and DR clipping.
PARAM_BOUNDS = {
    "mass": (0.3, 3.0),
    "length": (0.3, 3.0),
}

# Fixed physics constants shared by the real env and every simulator.
GRAVITY = 10.0
DT = 0.05
MAX_TORQUE = 2.0
MAX_SPEED = 8.0

# --------------------------------------------------------------------------- #
# Default training / evaluation hyperparameters (overridable from CLI)
# --------------------------------------------------------------------------- #
@dataclass
class HParams:
    algo: str = "sac"               # "sac" (sample-efficient; solves Pendulum) or "ppo"
    total_timesteps: int = 75_000   # bumped: the harder theta*=(2.0,1.5) env needs a
    #                                 bit more than the easy case to fully converge
    n_eval_episodes: int = 20       # zero-shot eval episodes on the real system
    id_rollout_len: int = 200       # steps per real rollout used for system ID
    id_mode: str = "grid"           # "grid" (real least-squares ID) or "ideal"
    id_grid_res: int = 61           # resolution of the (mass,length) ID grid search
    id_obs_noise: float = 1.0       # sensor noise (std) on measured angular velocity;
    #                                 makes ID error shrink ~1/sqrt(n) so the real-data
    #                                 budget axis is non-degenerate (noise-free ID is
    #                                 trivially solved with a handful of rollouts).
    # Learner hyperparameters
    learning_rate: float = 1e-3     # good for SAC on Pendulum (SB3 zoo value)
    batch_size: int = 256
    gamma: float = 0.99
    # SAC-specific
    buffer_size: int = 100_000
    learning_starts: int = 1000
    train_freq: int = 1
    gradient_steps: int = 1
    tau: float = 0.005
    # PPO-specific (only used when algo == "ppo")
    n_steps: int = 1024
    ent_coef: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Task-id <-> (n, w, seed) mapping
# --------------------------------------------------------------------------- #
def total_jobs() -> int:
    return len(N_GRID) * len(W_GRID) * len(SEEDS)


def decode_task_id(task_id: int) -> Tuple[int, float, int]:
    """Map a flat array index to (n, w, seed).

    Layout (most-significant first): n, then w, then seed.
    """
    if not (0 <= task_id < total_jobs()):
        raise ValueError(
            f"task_id {task_id} out of range [0, {total_jobs()})"
        )
    n_w_seed = len(W_GRID) * len(SEEDS)
    n_idx = task_id // n_w_seed
    rem = task_id % n_w_seed
    w_idx = rem // len(SEEDS)
    seed_idx = rem % len(SEEDS)
    return N_GRID[n_idx], W_GRID[w_idx], SEEDS[seed_idx]


def run_name(n: int, w: float, seed: int) -> str:
    """Deterministic, filesystem-safe directory name for a run."""
    return f"n{n}_w{w:g}_seed{seed}"


if __name__ == "__main__":
    # Print the grid so a launcher can verify the expected job count.
    print(f"TOTAL_JOBS={total_jobs()}")
    print(f"N_GRID={N_GRID}")
    print(f"W_GRID={W_GRID}")
    print(f"SEEDS={SEEDS}")
