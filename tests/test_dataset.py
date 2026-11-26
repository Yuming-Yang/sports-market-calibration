from datetime import UTC, datetime

import pandas as pd
import pytest
from fakes import event

from smcal.dataset import LOOKBACK_MIN, build_table
from smcal.results import find_result


def results_frame(rows: list[tuple]) -> pd.DataFrame:
    df = pd.DataFrame(
        rows,
        columns=["date", "home_team", "away_team", "home_score", "away_score", "neutral"],
    )
    df["date"] = pd.to_datetime(df["date"])
    df["tournament"] = "FIFA World Cup"
    df["host"] = df["home_team"].where(~df["neutral"])
    return df


KICKOFF = int(datetime(2026, 7, 19, 19, tzinfo=UTC).timestamp())


def flat_prices(levels: dict[str, float], calls: list | None = None):
    def fetch(token: str, start: int, end: int) -> dict:
        if calls is not None:
            calls.append((token, start, end))
        p = levels[token[-1]]
        return {"history": [{"t": t, "p": p} for t in range(start, end + 1, 60)]}

    return fetch


def test_build_table_prices_and_outcome() -> None:
    res = results_frame([("2026-07-19", "Spain", "Argentina", 1, 0, True)])
    calls: list = []
    table, excluded = build_table(
        [event(winner="D")], res, flat_prices({"H": 0.4, "D": 0.3, "A": 0.32}, calls)
    )
    assert excluded == []
    row = table.iloc[0]
    assert row["p_home"] == pytest.approx(0.4 / 1.02)
    assert row["raw_sum"] == pytest.approx(1.02)
    # Full time 1-0 but the 90-minute market resolved draw: extra time is inferred.
    assert row["outcome"] == "D" and bool(row["extra_time"])
    assert not row["thin"]
    # Every request ends at kickoff: nothing after kickoff is ever read.
    assert {(s, e) for _, s, e in calls} == {(KICKOFF - 60 * LOOKBACK_MIN, KICKOFF)}


def test_market_win_against_a_drawn_full_time_score_is_excluded() -> None:
    # Extra time can turn a draw into a win, never a win into a draw.
    res = results_frame([("2026-07-19", "Spain", "Argentina", 1, 1, True)])
    table, excluded = build_table(
        [event(winner="H")], res, flat_prices({"H": 0.4, "D": 0.3, "A": 0.3})
    )
    assert table.empty
    assert "full-time score" in excluded[0].reason


def test_missing_result_is_listed() -> None:
    res = results_frame([("2026-07-01", "Spain", "Argentina", 1, 1, True)])
    _, excluded = build_table([event()], res, flat_prices({"H": 0.4, "D": 0.3, "A": 0.3}))
    assert excluded[0].slug == "fifwc-spa-arg-2026-07-19"
    assert "0 result rows" in excluded[0].reason


def test_find_result_swaps_teams_and_keeps_host() -> None:
    res = results_frame([("2026-06-30", "Mexico", "Ecuador", 2, 0, False)])
    row = find_result(res, "Ecuador", "Mexico", pd.Timestamp("2026-07-01"))
    assert (row["home_team"], row["home_score"], row["away_score"]) == ("Ecuador", 0, 2)
    assert row["host"] == "Mexico"


def test_find_result_maps_polymarket_names() -> None:
    res = results_frame([("2026-06-11", "South Korea", "Czech Republic", 2, 1, True)])
    row = find_result(res, "Korea Republic", "Czechia", pd.Timestamp("2026-06-11"))
    assert row["home_score"] == 2
