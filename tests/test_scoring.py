import numpy as np
import pytest

from smcal.scoring import brier, brier_each, log_loss, rps_each


def test_hand_computed_scores() -> None:
    p = np.array([[0.5, 0.3, 0.2], [0.5, 0.3, 0.2]])
    y = np.array([0, 2])
    # Home win: (0.5)^2 + 0.3^2 + 0.2^2 = 0.38. Away win: 0.5^2 + 0.3^2 + 0.8^2 = 0.98.
    assert brier_each(p, y) == pytest.approx([0.38, 0.98])
    # RPS, home win: cumulative (0.5, 0.8) v (1, 1) -> (0.25 + 0.04) / 2 = 0.145.
    # Away win: (0.5, 0.8) v (0, 0) -> (0.25 + 0.64) / 2 = 0.445.
    assert rps_each(p, y) == pytest.approx([0.145, 0.445])
    assert log_loss(p, y) == pytest.approx(-(np.log(0.5) + np.log(0.2)) / 2)


def test_rps_prefers_a_near_miss() -> None:
    # Outcome draw. Forecast A leans home, B leans away by the same amount: equal RPS.
    # Forecast C puts the same mass on the far outcome when the result is a home win.
    a = np.array([[0.6, 0.2, 0.2]])
    c = np.array([[0.2, 0.2, 0.6]])
    near = rps_each(a, np.array([1]))[0]
    far = rps_each(c, np.array([0]))[0]
    assert near < far


def test_perfect_forecast_scores_zero() -> None:
    p = np.eye(3)
    y = np.array([0, 1, 2])
    assert brier(p, y) == 0 and rps_each(p, y).max() == 0


def test_invalid_forecasts_are_rejected() -> None:
    with pytest.raises(ValueError):
        brier(np.array([[0.5, 0.5, 0.5]]), np.array([0]))
    with pytest.raises(ValueError):
        brier(np.array([[0.5, 0.5, 0.0]]), np.array([3]))
