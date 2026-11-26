import numpy as np
import pandas as pd

from smcal.forecast import world_cup_forecasts
from smcal.models.elo import EloParams
from smcal.models.poisson import FitConfig


def history() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    teams = ["A", "B", "C", "D"]
    rows = []
    for d in pd.date_range("2022-01-01", periods=200, freq="7D"):
        h, a = rng.choice(teams, 2, replace=False)
        tour = "FIFA World Cup" if d.year == 2022 and d.month == 12 else "Friendly"
        if d.month % 2:
            tour = "FIFA World Cup qualification"
        rows.append((d, h, a, int(rng.poisson(1.4)), int(rng.poisson(1.1)), tour, None))
    # Two World Cup matches on consecutive days; the second is played after the first.
    rows.append((pd.Timestamp("2026-06-20"), "A", "B", 1, 0, "FIFA World Cup", None))
    rows.append((pd.Timestamp("2026-06-21"), "C", "A", 0, 0, "FIFA World Cup", None))
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
    return df.sort_values("date", kind="stable").reset_index(drop=True)


def table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "slug": ["m1", "m2"],
            # m1 kicks off at 01:00 UTC on the 21st: local date the 20th.
            "kickoff": ["2026-06-21T01:00:00+00:00", "2026-06-21T19:00:00+00:00"],
            "home": ["A", "C"],
            "away": ["B", "A"],
            "host": [None, None],
        }
    )


def test_no_model_input_is_dated_on_or_after_the_match() -> None:
    """Changing a match's own result, or anything later, must not change its forecast."""
    res = history()
    base = world_cup_forecasts(res, table(), EloParams(), FitConfig(window_years=2))
    poisoned = res.copy()
    late = poisoned["date"] >= pd.Timestamp("2026-06-20")
    poisoned.loc[late, ["home_score", "away_score"]] = [[9, 0]] * int(late.sum())
    again = world_cup_forecasts(poisoned, table(), EloParams(), FitConfig(window_years=2))
    cols = [c for c in base.columns if c.endswith(("_home", "_draw", "_away"))]
    m1 = base["slug"] == "m1"
    assert np.allclose(base.loc[m1, cols], again.loc[m1, cols])
    # m2 is the next day, so m1's result reaches it: the poisoning has an effect.
    assert not np.allclose(base.loc[~m1, cols], again.loc[~m1, cols])
    assert base.loc[m1, "model_cutoff"].iloc[0] == pd.Timestamp("2026-06-20")
    assert base.loc[~m1, "model_cutoff"].iloc[0] == pd.Timestamp("2026-06-21")


def test_forecasts_are_distributions() -> None:
    fc = world_cup_forecasts(history(), table(), EloParams(), FitConfig(window_years=2))
    for name in ("elo", "poisson", "dc", "base"):
        p = fc[[f"{name}_home", f"{name}_draw", f"{name}_away"]].to_numpy()
        assert np.allclose(p.sum(axis=1), 1)
