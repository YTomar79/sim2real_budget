"""System identification: estimate (mass, length) from n real-world rollouts.

Two modes:

  "grid"  (default, the real thing): collect `n` rollouts on the real system with
          a random policy, recover one-step transitions (theta, thetadot, u ->
          thetadot'), and grid-search (mass, length) to minimize one-step
          angular-velocity prediction error under the known Pendulum dynamics.
          More rollouts -> more transitions -> lower estimation error.

  "ideal" (fast ablation): bypass data collection and return
          theta_hat = theta_star + Gaussian noise whose scale decays as 1/sqrt(n).
          Lets you study the budget-allocation question without spending time
          debugging identification. Defensible for a 2-page preliminary result.

n == 0 in either mode -> no identification -> return the nominal PRIOR_THETA.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from . import config
from .param_pendulum import ParamPendulumEnv, make_real_env


def _collect_transitions(n_rollouts: int, rollout_len: int, seed: int,
                         obs_noise: float = 0.0):
    """Roll out a random policy on the REAL system; return transition arrays.

    `obs_noise` adds Gaussian measurement noise (std) to the recorded angular
    velocities, modeling a noisy sensor. With noise, more rollouts average the
    noise down, so the identification error decreases ~1/sqrt(n) and the
    real-data budget axis becomes meaningful.
    """
    env = make_real_env(max_episode_steps=rollout_len)
    rng = np.random.default_rng(seed)
    thetas, thetadots, us, next_thetadots = [], [], [], []
    for r in range(n_rollouts):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        for _ in range(rollout_len):
            u = rng.uniform(-config.MAX_TORQUE, config.MAX_TORQUE)
            theta = np.arctan2(obs[1], obs[0])
            thetadot = obs[2] + rng.normal(0.0, obs_noise)
            obs, _, _, trunc, _ = env.step(np.array([u], dtype=np.float32))
            thetas.append(theta)
            thetadots.append(thetadot)
            us.append(u)
            next_thetadots.append(obs[2] + rng.normal(0.0, obs_noise))
            if trunc:
                break
    return (
        np.asarray(thetas),
        np.asarray(thetadots),
        np.asarray(us),
        np.asarray(next_thetadots),
    )


def _predict_next_thetadot(theta, thetadot, u, mass, length):
    g, dt = config.GRAVITY, config.DT
    nd = thetadot + (
        3 * g / (2 * length) * np.sin(theta) + 3.0 / (mass * length**2) * u
    ) * dt
    return np.clip(nd, -config.MAX_SPEED, config.MAX_SPEED)


def _grid_identify(transitions, grid_res: int) -> Tuple[float, float]:
    thetas, thetadots, us, next_thetadots = transitions
    m_lo, m_hi = config.PARAM_BOUNDS["mass"]
    l_lo, l_hi = config.PARAM_BOUNDS["length"]
    m_cands = np.linspace(m_lo, m_hi, grid_res)
    l_cands = np.linspace(l_lo, l_hi, grid_res)

    best = (config.PRIOR_THETA[0], config.PRIOR_THETA[1])
    best_mse = np.inf
    for m in m_cands:
        for l in l_cands:
            pred = _predict_next_thetadot(thetas, thetadots, us, m, l)
            mse = float(np.mean((pred - next_thetadots) ** 2))
            if mse < best_mse:
                best_mse = mse
                best = (float(m), float(l))
    return best


def identify(
    n_rollouts: int,
    hp: config.HParams,
    seed: int,
) -> Tuple[Tuple[float, float], float]:
    """Return (theta_hat, id_error) where id_error = ||theta_hat - theta_star||."""
    theta_star = np.asarray(config.THETA_STAR)

    if n_rollouts <= 0:
        theta_hat = tuple(float(x) for x in config.PRIOR_THETA)
        err = float(np.linalg.norm(np.asarray(theta_hat) - theta_star))
        return theta_hat, err

    if hp.id_mode == "ideal":
        rng = np.random.default_rng(seed + 10_000)
        scale = 0.5 / np.sqrt(n_rollouts)  # error shrinks ~1/sqrt(n)
        noise = rng.normal(0.0, scale, size=2)
        m = float(np.clip(theta_star[0] + noise[0], *config.PARAM_BOUNDS["mass"]))
        l = float(np.clip(theta_star[1] + noise[1], *config.PARAM_BOUNDS["length"]))
        theta_hat = (m, l)
    elif hp.id_mode == "grid":
        transitions = _collect_transitions(
            n_rollouts, hp.id_rollout_len, seed, obs_noise=hp.id_obs_noise
        )
        theta_hat = _grid_identify(transitions, hp.id_grid_res)
    else:
        raise ValueError(f"unknown id_mode '{hp.id_mode}'")

    err = float(np.linalg.norm(np.asarray(theta_hat) - theta_star))
    return theta_hat, err
