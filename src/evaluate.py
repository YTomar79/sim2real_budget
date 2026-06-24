"""Zero-shot evaluation of a trained policy on the REAL system (theta_star)."""
from __future__ import annotations

from typing import Tuple

import numpy as np

from .param_pendulum import make_real_env


def evaluate_on_real(
    model,
    n_eval_episodes: int,
    seed: int,
    max_episode_steps: int = 200,
) -> Tuple[float, float, list]:
    """Run the deterministic policy on the real env; return (mean, std, returns)."""
    env = make_real_env(max_episode_steps=max_episode_steps)
    returns = []
    for ep in range(n_eval_episodes):
        obs, _ = env.reset(seed=seed + 50_000 + ep)
        done = False
        total = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            done = terminated or truncated
        returns.append(total)
    returns = np.asarray(returns)
    return float(returns.mean()), float(returns.std()), returns.tolist()
