import numpy as np
import pandas as pd
import pytest

from smcal.models.elo import (
    EloParams,
    OrderedLogit,
    fit_ordered_logit,
    goal_multiplier,
    k_factor,
    run_elo,
)


def frame(rows):
    df = pd.DataFrame(
        rows,
        columns=[
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "tournament",
            "host",
        ],
    )
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_k_table_and_goal_multiplier() -> None:
    assert k_factor("FIFA World Cup") == 60
    assert k_factor("Copa América") == 50
    assert k_factor("FIFA World Cup qualification") == 40
    assert k_factor("UEFA Nations League") == 40
    assert k_factor("COSAFA Cup") == 30
    assert k_factor("Friendly") == 20
    assert [goal_multiplier(g) for g in (0, 1, -2, 3, 5)] == [1, 1, 1.5, 1.75, 2.0]


def test_hand_computed_update_with_home_advantage() -> None:
    # A hosts B in a friendly, both at 1500. dr = +100, expected = 1/(10^-0.25 + 1) = 0.640.
    # A wins 2-0: delta = 20 * 1.5 * (1 - 0.640) = 10.80. The next day A is rated 1510.80,
    # B 1489.20; at a neutral venue the difference is 21.60.
    df = frame(
        [
            ("2020-01-01", "A", "B", 2, 0, "Friendly", "A"),
            ("2020-01-02", "A", "B", 0, 0, "Friendly", None),
        ]
    )
    diffs, ratings = run_elo(df, EloParams(home_adv=100))
    we = 1 / (10 ** (-100 / 400) + 1)
    delta = 20 * 1.5 * (1 - we)
    assert diffs[0] == pytest.approx(100)
    assert diffs[1] == pytest.approx(2 * delta)
    assert ratings["A"] + ratings["B"] == pytest.approx(3000)


def test_same_day_results_are_not_used() -> None:
    # B's big win against C on the same day must not change B's rating for its match v A.
    base = [("2020-01-01", "A", "B", 1, 1, "Friendly", None)]
    other = ("2020-01-01", "B", "C", 9, 0, "Friendly", None)
    d1, _ = run_elo(frame(base), EloParams())
    d2, _ = run_elo(frame([*base, other]), EloParams())
    assert d1[0] == d2[0] == 0.0


def test_unsorted_input_is_rejected() -> None:
    df = frame(
        [
            ("2020-01-02", "A", "B", 1, 0, "Friendly", None),
            ("2020-01-01", "A", "B", 1, 0, "Friendly", None),
        ]
    )
    with pytest.raises(ValueError, match="sorted"):
        run_elo(df, EloParams())


def test_ordered_logit_recovers_parameters() -> None:
    rng = np.random.default_rng(0)
    true = OrderedLogit(slope=0.45, cut=0.55)
    dr = rng.normal(0, 250, 40_000)
    p = true.probs(dr)
    u = rng.random(len(dr))[:, None]
    y = (u > np.cumsum(p, axis=1)).sum(axis=1)
    est = fit_ordered_logit(dr, y)
    assert est.slope == pytest.approx(true.slope, abs=0.02)
    assert est.cut == pytest.approx(true.cut, abs=0.02)


def test_ordered_logit_is_symmetric_and_valid() -> None:
    m = OrderedLogit(0.5, 0.4)
    p = m.probs([150.0, -150.0, 0.0])
    assert np.allclose(p.sum(axis=1), 1)
    assert (p > 0).all()
    assert p[0] == pytest.approx(p[1][::-1])
    assert p[2][0] == pytest.approx(p[2][2])
