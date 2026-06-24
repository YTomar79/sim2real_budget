"""Domain-randomization wrapper.

On every reset, resample the pendulum's (mass, length) from a uniform window of
relative half-width `w` centered on the point estimate `theta_hat`:

    mass   ~ U(m_hat * (1 - w), m_hat * (1 + w))
    length ~ U(l_hat * (1 - w), l_hat * (1 + w))

w = 0   -> always train at the point estimate (no breadth; pure fidelity reliance)
w large -> wide randomization (robust but suboptimal; the "breadth" lever)

Samples are clipped to physically plausible bounds from config.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import gymnasium as gym

from . import config
from .param_pendulum import ParamPendulumEnv


class DomainRandomizationWrapper(gym.Wrapper):
    def __init__(
        self,
        env: ParamPendulumEnv,
        theta_hat: Tuple[float, float],
        width: float,
        seed: Optional[int] = None,
        mode: str = "relative",
    ):
        super().__init__(env)
        self.m_hat, self.l_hat = float(theta_hat[0]), float(theta_hat[1])
        self.width = float(width)
        self.mode = mode
        self._rng = np.random.default_rng(seed)
        self._m_bounds = config.PARAM_BOUNDS["mass"]
        self._l_bounds = config.PARAM_BOUNDS["length"]

    def _sample_params(self) -> Tuple[float, float]:
        if self.mode == "prior":
            # Ignore the estimate: sample uniformly over the full plausible range
            # (a wide prior that brackets the truth). Width is unused here.
            m = self._rng.uniform(*self._m_bounds)
            l = self._rng.uniform(*self._l_bounds)
            return float(m), float(l)
        if self.width <= 0.0:
            return self.m_hat, self.l_hat
        m = self._rng.uniform(self.m_hat * (1 - self.width), self.m_hat * (1 + self.width))
        l = self._rng.uniform(self.l_hat * (1 - self.width), self.l_hat * (1 + self.width))
        m = float(np.clip(m, *self._m_bounds))
        l = float(np.clip(l, *self._l_bounds))
        return m, l

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        m, l = self._sample_params()
        self.env.set_params(m, l)
        return self.env.reset(seed=seed, options=options)
