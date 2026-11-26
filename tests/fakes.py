"""Hand-made API responses shaped like Polymarket's Gamma events. Prices are invented."""

from __future__ import annotations

import json
from typing import Any

DESC_90 = (
    'If Spain wins, this market will resolve to "Yes".\n'
    "This market refers only to the outcome within the first 90 minutes of regular "
    "play plus stoppage time."
)


def market(question: str, token: str, yes_won: bool, kickoff: str, desc: str = DESC_90):
    return {
        "question": question,
        "outcomes": json.dumps(["Yes", "No"]),
        "outcomePrices": json.dumps(["1", "0"] if yes_won else ["0", "1"]),
        "clobTokenIds": json.dumps([token, token + "-no"]),
        "gameStartTime": kickoff,
        "description": desc,
    }


def event(
    home: str = "Spain",
    away: str = "Argentina",
    date: str = "2026-07-19",
    kickoff: str = "2026-07-19 19:00:00+00",
    winner: str = "D",
    slug: str | None = None,
) -> dict[str, Any]:
    slug = slug or f"fifwc-{home[:3].lower()}-{away[:3].lower()}-{date}"
    return {
        "slug": slug,
        "teams": [
            {"name": home, "ordering": "home"},
            {"name": away, "ordering": "away"},
        ],
        "markets": [
            market(f"Will {home} win on {date}?", f"{slug}-H", winner == "H", kickoff),
            market(f"Will {home} vs. {away} end in a draw?", f"{slug}-D", winner == "D", kickoff),
            market(f"Will {away} win on {date}?", f"{slug}-A", winner == "A", kickoff),
        ],
    }
