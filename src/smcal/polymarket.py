"""Read-only access to Polymarket's public Gamma and CLOB endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from smcal.http import get_json

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
WORLD_CUP_SERIES_ID = 11433  # series slug "soccer-fifwc"
PAGE = 50  # Gamma serves larger pages gzip-only; 50 keeps responses small


def series_events(series_id: int, cache_dir: Path) -> list[dict[str, Any]]:
    """All events in a series, paging until an empty page."""
    events: dict[str, dict[str, Any]] = {}
    offset = 0
    while True:
        url = f"{GAMMA}/events?series_id={series_id}&limit={PAGE}&offset={offset}"
        page = get_json(url, cache_dir)
        if not isinstance(page, list):
            raise RuntimeError(f"unexpected response from {url}: {str(page)[:200]}")
        if not page:
            break
        for e in page:
            events[e["id"]] = e
        offset += PAGE
    return sorted(events.values(), key=lambda e: e["slug"])


def price_history(token_id: str, start_ts: int, end_ts: int, cache_dir: Path) -> dict[str, Any]:
    """One-minute price history for a token between two Unix times.

    Explicit bounds are required: `interval=max` returned nothing for settled markets
    when checked on 2026-11-15.
    """
    url = f"{CLOB}/prices-history?market={token_id}&startTs={start_ts}&endTs={end_ts}&fidelity=1"
    return get_json(url, cache_dir)
