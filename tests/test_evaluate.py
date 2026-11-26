import numpy as np
import pytest

from smcal.evaluate import (
    bootstrap_indices,
    brier_decomposition,
    brier_decomposition_binary,
    calibration_exponent,
    mean_ci,
    paired_difference,
    reliability_table,
    wilson,
)
from smcal.scoring import brier


def random_forecasts(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = rng.dirichlet([2, 1, 1.5], size=n)
    y = np.array([rng.choice(3, p=row) for row in p])
    return p, y


def test_decomposition_sums_to_brier_exactly() -> None:
    p, y = random_forecasts(500, 0)
    for bins in (3, 10, 25):
        parts = brier_decomposition(p, y, n_bins=bins)
        assert parts.total == pytest.approx(brier(p, y), abs=1e-12)


def test_decomposition_hand_computed() -> None:
    # Four forecasts, two bins at 0.5. Bin [0, .5): f = 0.2, 0.4 with o = 0, 1.
    # Bin [.5, 1]: f = 0.8, 0.8 with o = 1, 1.
    f = np.array([0.2, 0.4, 0.8, 0.8])
    o = np.array([0, 1, 1, 1])
    d = brier_decomposition_binary(f, o, n_bins=2)
    # bin means: (0.3, 0.5) and (0.8, 1.0); obar = 0.75
    assert d.reliability == pytest.approx((2 * 0.2**2 + 2 * 0.2**2) / 4)
    assert d.resolution == pytest.approx((2 * 0.25**2 + 2 * 0.25**2) / 4)
    assert d.uncertainty == pytest.approx(0.75 * 0.25)
    assert d.within_bin_variance == pytest.approx((0.1**2 + 0.1**2) / 4)
    assert d.within_bin_covariance == pytest.approx(2 * (-0.5 * -0.1 + 0.5 * 0.1) / 4)
    assert d.total == pytest.approx(np.mean((f - o) ** 2))


def test_wilson_known_value() -> None:
    lo, hi = wilson(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-4)
    assert hi == pytest.approx(0.7634, abs=1e-4)


def test_paired_bootstrap_of_identical_forecasters_is_zero() -> None:
    s = np.random.default_rng(1).random(100)
    d = paired_difference(s, s, bootstrap_indices(100, 500, 0))
    assert d.diff == d.lo == d.hi == 0.0


def test_pairing_narrows_the_interval() -> None:
    rng = np.random.default_rng(2)
    common = rng.normal(0, 1, 200)  # match difficulty, shared by both forecasters
    a = common + rng.normal(0, 0.1, 200)
    b = common + 0.05 + rng.normal(0, 0.1, 200)
    idx = bootstrap_indices(200, 2000, 3)
    d = paired_difference(a, b, idx)
    _, lo_a, hi_a = mean_ci(a, idx)
    assert (d.hi - d.lo) < (hi_a - lo_a) / 5
    assert d.hi < 0  # A really is better by 0.05 per match


def test_calibration_exponent_recovers_a_known_distortion() -> None:
    rng = np.random.default_rng(4)
    truth = rng.dirichlet([2, 1, 1.5], size=20_000)
    y = (rng.random((len(truth), 1)) > np.cumsum(truth, axis=1)).sum(axis=1)
    # A forecaster that flattens the truth: forecast = truth ** (1 / 1.3), normalized.
    flat = truth ** (1 / 1.3)
    flat /= flat.sum(axis=1, keepdims=True)
    assert calibration_exponent(flat, y) == pytest.approx(1.3, abs=0.05)
    assert calibration_exponent(truth, y) == pytest.approx(1.0, abs=0.05)


def test_reliability_table_counts_every_forecast_once() -> None:
    p, y = random_forecasts(50, 5)
    rows = reliability_table(p, y, n_bins=5)
    assert sum(r["n"] for r in rows) == 150
