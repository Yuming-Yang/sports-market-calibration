"""Build the per-match table: pre-match market probabilities and the 90-minute outcome."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from smcal.markets import OUTCOMES, Excluded, MatchMarket, is_match_event, parse_event
from smcal.prematch import WindowRule, normalize, twap
from smcal.results import find_result, full_time_outcome

# How far before kickoff to request prices: enough to find the price in force at the
# window start even if the market was quiet for a while.
LOOKBACK_MIN = 360

DEFAULT_RULE = WindowRule()

PriceFetcher = Callable[[str, int, int], dict[str, Any]]


@dataclass(frozen=True)
class Exclusion:
    slug: str
    reason: str


def check_outcome(market: MatchMarket, full_time: str) -> bool:
    """Check the market's 90-minute resolution against the full-time score.

    They must agree unless the match went to extra time, which can only turn a 90-minute
    draw into a win. Returns True when extra time is inferred.
    """
    if market.resolved == full_time:
        return False
    if market.resolved == "D":
        return True
    raise Excluded(f"market resolved {market.resolved} but the full-time score gives {full_time}")


def build_table(
    events: Iterable[dict[str, Any]],
    results: pd.DataFrame,
    fetch_prices: PriceFetcher,
    rule: WindowRule = DEFAULT_RULE,
) -> tuple[pd.DataFrame, list[Exclusion]]:
    rows: list[dict[str, Any]] = []
    excluded: list[Exclusion] = []
    for event in events:
        if not is_match_event(event):
            continue
        slug = event["slug"]
        try:
            m = parse_event(event)
            res = find_result(results, m.home, m.away, pd.Timestamp(m.kickoff))
            ft = full_time_outcome(int(res["home_score"]), int(res["away_score"]))
            extra_time = check_outcome(m, ft)

            k = int(m.kickoff.timestamp())
            start, end = k - 60 * rule.start_min, k - 60 * rule.end_min
            raw, info = {}, {}
            for o in OUTCOMES:
                hist = fetch_prices(m.tokens[o], k - 60 * LOOKBACK_MIN, k)["history"]
                try:
                    w = twap([h["t"] for h in hist], [h["p"] for h in hist], start, end)
                except ValueError as err:
                    raise Excluded(f"outcome {o}: {err}") from err
                raw[o] = w.twap
                info[o] = w
            probs, raw_sum = normalize(raw)
        except (Excluded, LookupError) as err:
            excluded.append(Exclusion(slug, str(err)))
            continue

        rows.append(
            {
                "slug": slug,
                "kickoff": m.kickoff,
                "home": res["home_team"],
                "away": res["away_team"],
                "host": res["host"] if isinstance(res["host"], str) else None,
                "home_goals_ft": int(res["home_score"]),
                "away_goals_ft": int(res["away_score"]),
                "extra_time": extra_time,
                "outcome": m.resolved,
                "p_home": probs["H"],
                "p_draw": probs["D"],
                "p_away": probs["A"],
                "raw_sum": raw_sum,
                "min_points": min(w.n_points for w in info.values()),
                "min_changes": min(w.n_changes for w in info.values()),
                "max_staleness_s": max(w.staleness_s for w in info.values()),
                "thin": min(w.n_points for w in info.values()) < rule.min_points,
            }
        )
    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values("kickoff", kind="stable").reset_index(drop=True)
    return table, excluded
