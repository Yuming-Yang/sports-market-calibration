import pytest

from smcal.prematch import WindowRule, normalize, twap


def test_twap_hand_computed() -> None:
    # Window [0, 100]. Price 0.40 in force at the start (set at t=-10), 0.50 from t=25,
    # 0.60 from t=75. Average = (0.40*25 + 0.50*50 + 0.60*25) / 100 = 0.50.
    w = twap([-10, 25, 75, 150], [0.40, 0.50, 0.60, 0.90], start=0, end=100)
    assert w.twap == pytest.approx(0.50)
    assert w.n_points == 2
    assert w.n_changes == 2
    assert w.staleness_s == 10


def test_prices_after_the_window_end_are_ignored() -> None:
    # A move at kickoff (news on line-ups) must not reach the pre-match price.
    base = twap([0, 50], [0.3, 0.3], start=0, end=100)
    moved = twap([0, 50, 101], [0.3, 0.3, 0.9], start=0, end=100)
    assert base.twap == moved.twap == pytest.approx(0.3)


def test_observation_exactly_at_window_end_has_zero_weight() -> None:
    w = twap([0, 100], [0.2, 0.8], start=0, end=100)
    assert w.twap == pytest.approx(0.2)


def test_no_price_before_window_start_raises() -> None:
    with pytest.raises(ValueError, match="no price"):
        twap([10, 20], [0.5, 0.5], start=0, end=100)


def test_window_rule_must_end_before_kickoff() -> None:
    assert WindowRule().start_min == 120 and WindowRule().end_min == 60
    with pytest.raises(ValueError):
        WindowRule(start_min=60, end_min=120)
    with pytest.raises(ValueError):
        WindowRule(start_min=10, end_min=-5)


def test_normalize_keeps_raw_sum() -> None:
    p, total = normalize({"H": 0.5, "D": 0.3, "A": 0.25})
    assert total == pytest.approx(1.05)
    assert sum(p.values()) == pytest.approx(1.0)
    assert p["H"] == pytest.approx(0.5 / 1.05)
