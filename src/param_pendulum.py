"""Parametric Pendulum environment with settable physical parameters.

This is a self-contained reimplementation of the classic Gymnasium Pendulum-v1
dynamics, with the pendulum **mass** and **length** exposed as settable
parameters (`set_params`). Everything else (gravity, dt, torque/speed limits,
reward) matches Pendulum-v1 so behavior is familiar and SAC solves it quickly.

We need our own env because Gymnasium's Pendulum only exposes `g`, not mass and
length, and the whole experiment hinges on a sim/real mismatch in those two
parameters.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from . import config


def angle_normalize(x: np.ndarray | float):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


class ParamPendulumEnv(gym.Env):
    """Pendulum with adjustable (mass, length)."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        mass: float = 1.0,
        length: float = 1.0,
        max_episode_steps: int = 200,
    ):
        super().__init__()
        self.mass = float(mass)
        self.length = float(length)
        self.g = config.GRAVITY
        self.dt = config.DT
        self.max_torque = config.MAX_TORQUE
        self.max_speed = config.MAX_SPEED
        self.max_episode_steps = max_episode_steps
        self._step = 0

        high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )
        self.state: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    def set_params(self, mass: float, length: float) -> None:
        """Update the physical parameters in place (used by the DR wrapper)."""
        self.mass = float(mass)
        self.length = float(length)

    def _get_obs(self) -> np.ndarray:
        theta, thetadot = self.state
        return np.array(
            [np.cos(theta), np.sin(theta), thetadot], dtype=np.float32
        )

    # ------------------------------------------------------------------ #
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        high = np.array([np.pi, 1.0])
        self.state = self.np_random.uniform(low=-high, high=high)
        self._step = 0
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        theta, thetadot = self.state
        u = float(np.clip(action, -self.max_torque, self.max_torque)[0])

        g, m, l, dt = self.g, self.mass, self.length, self.dt

        cost = angle_normalize(theta) ** 2 + 0.1 * thetadot**2 + 0.001 * (u**2)

        new_thetadot = thetadot + (
            3 * g / (2 * l) * np.sin(theta) + 3.0 / (m * l**2) * u
        ) * dt
        new_thetadot = float(np.clip(new_thetadot, -self.max_speed, self.max_speed))
        new_theta = theta + new_thetadot * dt

        self.state = np.array([new_theta, new_thetadot])
        self._step += 1
        truncated = self._step >= self.max_episode_steps
        return self._get_obs(), -cost, False, truncated, {}


def make_real_env(max_episode_steps: int = 200) -> ParamPendulumEnv:
    """The held-out 'real' system at THETA_STAR."""
    m_star, l_star = config.THETA_STAR
    return ParamPendulumEnv(
        mass=m_star, length=l_star, max_episode_steps=max_episode_steps
    )
