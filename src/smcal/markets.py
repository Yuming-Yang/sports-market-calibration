"""Parse Polymarket match events into one record per match.

A World Cup match on Polymarket is an event with three binary markets: home win, draw,
away win. Each market's "Yes" token pays 1 if its outcome happens. This module turns the
Gamma API's event JSON into a `MatchMarket` and reads two things from the market text
that the analysis depends on: which result the market settles on, and how it resolved.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

OUTCOMES = ("H", "D", "A")

MATCH_SLUG = re.compile(r"^fifwc-[a-z]+-[a-z]+-\d{4}-\d{2}-\d{2}$")

NINETY_MINUTES = "first 90 minutes of regular play plus stoppage time"


class Excluded(Exception):
    """Raised when a match cannot be used. The message is the reason, listed in results."""


@dataclass(frozen=True)
class MatchMarket:
    slug: str
    kickoff: datetime
    home: str
    away: str
    tokens: dict[str, str]  # outcome -> Yes token id
    settles_on: str  # "90min"; other rules are excluded
    resolved: str  # "H", "D" or "A"


def is_match_event(event: dict[str, Any]) -> bool:
    return bool(MATCH_SLUG.match(event.get("slug", "")))


def settlement_rule(description: str) -> str:
    """Return the result a market settles on, read from its description.

    Every 2026 World Cup match market checked on 2026-11-26 settles on the 90-minute
    result, knockout matches included. A description that does not say so is not guessed
    at; the match is excluded.
    """
    text = " ".join(description.split())
    if NINETY_MINUTES in text:
        return "90min"
    raise Excluded("settlement rule not stated as the 90-minute result")


def _classify(question: str, home: str, away: str) -> str:
    q = question.strip()
    if q.endswith("end in a draw?"):
        return "D"
    if q.startswith(f"Will {home} win"):
        return "H"
    if q.startswith(f"Will {away} win"):
        return "A"
    raise Excluded(f"cannot classify market question {q!r}")


def _resolved_yes(market: dict[str, Any]) -> bool | None:
    outcomes = json.loads(market["outcomes"])
    prices = [float(p) for p in json.loads(market["outcomePrices"])]
    if outcomes != ["Yes", "No"]:
        raise Excluded(f"unexpected outcomes {outcomes}")
    if prices == [1.0, 0.0]:
        return True
    if prices == [0.0, 1.0]:
        return False
    return None


def parse_kickoff(text: str) -> datetime:
    """Parse Gamma's `gameStartTime`, for example "2026-07-19 19:00:00+00"."""
    dt = datetime.strptime(text.replace("+00", "+0000"), "%Y-%m-%d %H:%M:%S%z")
    return dt.astimezone(UTC)


def parse_event(event: dict[str, Any]) -> MatchMarket:
    teams = {t["ordering"]: t["name"] for t in event.get("teams", [])}
    if set(teams) != {"home", "away"}:
        raise Excluded("event does not name a home and an away team")
    home, away = teams["home"], teams["away"]

    markets = event.get("markets", [])
    if len(markets) != 3:
        raise Excluded(f"expected 3 markets, found {len(markets)}")

    tokens: dict[str, str] = {}
    winners: list[str] = []
    kickoffs: set[str] = set()
    rules: set[str] = set()
    for m in markets:
        outcome = _classify(m["question"], home, away)
        if outcome in tokens:
            raise Excluded(f"two markets for outcome {outcome}")
        tokens[outcome] = json.loads(m["clobTokenIds"])[0]
        rules.add(settlement_rule(m.get("description", "")))
        kickoffs.add(m["gameStartTime"])
        resolved = _resolved_yes(m)
        if resolved is None:
            raise Excluded("market not resolved to 0 or 1")
        if resolved:
            winners.append(outcome)

    if len(kickoffs) != 1:
        raise Excluded("markets disagree on kickoff time")
    if len(winners) != 1:
        raise Excluded(f"{len(winners)} markets resolved Yes")

    return MatchMarket(
        slug=event["slug"],
        kickoff=parse_kickoff(kickoffs.pop()),
        home=home,
        away=away,
        tokens=tokens,
        settles_on=rules.pop(),
        resolved=winners[0],
    )
