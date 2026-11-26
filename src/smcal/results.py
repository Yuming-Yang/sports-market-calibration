"""International match results (martj42/international_results, CC0-1.0).

The dataset records the score at the end of the match, after extra time when there was
any. Penalty shoot-outs are in a separate file and are not goals.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd

# Pinned so that every number can be rebuilt from the same file.
RESULTS_COMMIT = "ca84241bb3eed854a1956a2fab442fe4daad40dd"
RESULTS_URL = (
    f"https://raw.githubusercontent.com/martj42/international_results/{RESULTS_COMMIT}/results.csv"
)

# Polymarket team name -> results-dataset team name, where they differ.
POLYMARKET_TO_RESULTS = {
    "Cabo Verde": "Cape Verde",
    "Czechia": "Czech Republic",
    "Côte d'Ivoire": "Ivory Coast",
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "Türkiye": "Turkey",
}


def results_name(polymarket_name: str) -> str:
    return POLYMARKET_TO_RESULTS.get(polymarket_name, polymarket_name)


def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE")
    # The team playing at home, if any. Kept as its own column so that re-orienting a row
    # (see `find_result`) does not move home advantage to the wrong team.
    df["host"] = df["home_team"].where(~df["neutral"])
    return df.sort_values(["date", "home_team", "away_team"], kind="stable").reset_index(drop=True)


def full_time_outcome(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "H"
    if home_score < away_score:
        return "A"
    return "D"


def find_result(
    results: pd.DataFrame, home: str, away: str, kickoff_date: pd.Timestamp
) -> pd.Series:
    """Find the row for a match, oriented to the given home and away teams.

    Dates in the results file are local dates; Polymarket's kickoff is UTC. An evening
    match in the Americas can fall on the next UTC day, so a one-day tolerance is allowed.
    Returns the row with scores swapped if the file lists the teams the other way round.
    """
    h, a = results_name(home), results_name(away)
    day = pd.Timestamp(kickoff_date).normalize().tz_localize(None)
    near = results[(results["date"] - day).abs() <= timedelta(days=1)]
    same = near[(near["home_team"] == h) & (near["away_team"] == a)]
    swapped = near[(near["home_team"] == a) & (near["away_team"] == h)]
    if len(same) + len(swapped) != 1:
        raise LookupError(f"{len(same) + len(swapped)} result rows for {h} v {a} near {day:%F}")
    if len(same) == 1:
        return same.iloc[0]
    row = swapped.iloc[0].copy()
    row["home_team"], row["away_team"] = h, a
    row["home_score"], row["away_score"] = row["away_score"], row["home_score"]
    return row
