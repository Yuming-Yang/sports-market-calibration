"""Turn a price series into one pre-match probability per outcome.

The rule (SPEC section 2, decision D3): the time-weighted average price over a window that
ends before kickoff, by default from 120 to 60 minutes before. The price series is treated
as a step function: each observation holds until the next one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WindowRule:
    start_min: int = 120  # minutes before kickoff
    end_min: int = 60
    # Fewer observations than this inside the window flags the match as thin. Polymarket's
    # one-minute history has about 60 points in a 60-minute window when the feed is live.
    min_points: int = 30

    def __post_init__(self) -> None:
        if not self.start_min > self.end_min >= 0:
            raise ValueError("window must start before it ends, and end at or before kickoff")


@dataclass(frozen=True)
class WindowPrice:
    twap: float
    n_points: int  # observations inside (start, end]
    n_changes: int  # times the price changed inside the window
    staleness_s: int  # age of the price in force at window start


def twap(times: Sequence[int], prices: Sequence[float], start: int, end: int) -> WindowPrice:
    """Time-weighted average of a step-function price over [start, end].

    `times` are Unix seconds, ascending. The price in force at `start` is the last
    observation at or before `start`. If there is none, the window cannot be priced, and
    `ValueError` is raised: averaging only the part of the window after the first trade would
    quietly move the window closer to kickoff.
    """
    t = np.asarray(times, dtype=np.int64)
    p = np.asarray(prices, dtype=float)
    if t.shape != p.shape or t.ndim != 1:
        raise ValueError("times and prices must be 1-d and the same length")
    if t.size and np.any(np.diff(t) < 0):
        raise ValueError("times must be ascending")
    if end <= start:
        raise ValueError("end must be after start")

    first = np.searchsorted(t, start, side="right") - 1
    if first < 0:
        raise ValueError("no price at or before the window start")

    inside = (t > start) & (t <= end)
    # Breakpoints: window start, every observation inside, window end.
    knots = np.concatenate([[start], t[inside], [end]])
    levels = np.concatenate([[p[first]], p[inside]])
    durations = np.diff(knots)
    value = float(np.dot(levels, durations) / (end - start))

    window_levels = levels
    n_changes = int(np.count_nonzero(np.diff(window_levels) != 0))
    return WindowPrice(
        twap=value,
        n_points=int(inside.sum()),
        n_changes=n_changes,
        staleness_s=int(start - t[first]),
    )


def normalize(raw: dict[str, float]) -> tuple[dict[str, float], float]:
    """Divide the three outcome prices by their sum. Return the result and the raw sum."""
    total = sum(raw.values())
    if total <= 0:
        raise ValueError("prices sum to zero")
    return {k: v / total for k, v in raw.items()}, total
