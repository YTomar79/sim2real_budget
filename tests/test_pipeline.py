"""Fast, deterministic checks for the core pipeline invariants.

These run without training (no torch needed) and encode the properties the
experiment's conclusions depend on: a clean sweep grid, seed-determinism,
estimate-centered domain randomization, a fixed held-out system, and a
seed-paired bootstrap.
"""
import itertools

import numpy as np
import pytest

from src import config
from src.dr_wrapper import DomainRandomizationWrapper
from src.param_pendulum import ParamPendulumEnv, make_real_env
from src.system_id import identify
from src.stats import paired_bootstrap


# --------------------------------------------------------------------------- #
# Sweep grid: TOTAL_JOBS and the task_id <-> (n, w, seed) bijection
# --------------------------------------------------------------------------- #
def test_total_jobs_matches_grid():
    assert config.total_jobs() == len(config.N_GRID) * len(config.W_GRID) * len(config.SEEDS)


def test_decode_task_id_is_a_bijection_over_the_full_grid():
    decoded = [config.decode_task_id(i) for i in range(config.total_jobs())]
    assert len(set(decoded)) == config.total_jobs()                  # no duplicates
    full = set(itertools.product(config.N_GRID, config.W_GRID, config.SEEDS))
    assert set(decoded) == full                                      # full coverage


@pytest.mark.parametrize("bad", [-1, None])
def test_decode_task_id_rejects_out_of_range(bad):
    idx = config.total_jobs() if bad is None else bad
    with pytest.raises(ValueError):
        config.decode_task_id(idx)


# --------------------------------------------------------------------------- #
# Held-out "real" system is fixed and only (mass, length) vary
# --------------------------------------------------------------------------- #
def test_real_env_is_pinned_to_theta_star():
    env = make_real_env()
    assert (env.mass, env.length) == config.THETA_STAR


# --------------------------------------------------------------------------- #
# Domain randomization is centered on the ESTIMATE, not the true parameters
# --------------------------------------------------------------------------- #
def test_dr_width_zero_trains_at_the_point_estimate():
    theta_hat = (1.7, 1.3)
    env = DomainRandomizationWrapper(ParamPendulumEnv(), theta_hat=theta_hat, width=0.0, seed=0)
    env.reset(seed=0)
    assert (env.env.mass, env.env.length) == pytest.approx(theta_hat)


def test_dr_samples_are_centered_on_estimate_within_relative_band():
    theta_hat = (1.7, 1.3)              # deliberately != THETA_STAR (no oracle leak)
    width = 0.2
    env = DomainRandomizationWrapper(ParamPendulumEnv(), theta_hat=theta_hat, width=width, seed=0)
    for s in range(200):
        env.reset(seed=s)
        m, l = env.env.mass, env.env.length
        assert theta_hat[0] * (1 - width) - 1e-9 <= m <= theta_hat[0] * (1 + width) + 1e-9
        assert theta_hat[1] * (1 - width) - 1e-9 <= l <= theta_hat[1] * (1 + width) + 1e-9


# --------------------------------------------------------------------------- #
# System identification: prior fallback + seed-determinism
# --------------------------------------------------------------------------- #
def test_identify_with_zero_budget_returns_the_prior():
    theta_hat, err = identify(0, config.HParams(), seed=0)
    assert tuple(theta_hat) == pytest.approx(config.PRIOR_THETA)
    assert err == pytest.approx(float(np.linalg.norm(
        np.asarray(config.PRIOR_THETA) - np.asarray(config.THETA_STAR))))


def test_identify_is_seed_deterministic():
    hp = config.HParams(id_grid_res=21)        # small grid keeps the test fast
    a, _ = identify(3, hp, seed=7)
    b, _ = identify(3, hp, seed=7)
    assert a == b


# --------------------------------------------------------------------------- #
# Paired bootstrap: pairing by seed + fixed-RNG reproducibility
# --------------------------------------------------------------------------- #
def test_paired_bootstrap_requires_equal_length_paired_samples():
    with pytest.raises(AssertionError):
        paired_bootstrap(np.zeros(5), np.zeros(4))


def test_paired_bootstrap_is_reproducible_and_detects_a_real_gap():
    a = np.array([10.0, 11.0, 9.0, 12.0, 10.5])
    b = a - 5.0
    r1 = paired_bootstrap(a, b)
    r2 = paired_bootstrap(a, b)
    assert r1 == r2                       # fixed bootstrap RNG -> identical CIs
    assert r1["significant"] and r1["ci_lo"] > 0
    assert r1["mean_diff"] == pytest.approx(5.0)
