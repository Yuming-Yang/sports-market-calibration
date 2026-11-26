"""Climatological baseline: the floor any model must beat.

It knows the draw rate and nothing about the teams. Home and away win get equal shares of
the rest, because at a neutral venue the "home" label is arbitrary. The draw rate comes
from World Cup finals matches before the tournament being predicted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def draw_rate(results: pd.DataFrame, before: pd.Timestamp, since: str = "1998-01-01") -> float:
    wc = results[
        (results["tournament"] == "FIFA World Cup")
        & (results["date"] >= pd.Timestamp(since))
        & (results["date"] < pd.Timestamp(before))
    ]
    if wc.empty:
        raise ValueError("no World Cup matches in range")
    return float((wc["home_score"] == wc["away_score"]).mean())


def baseline_probs(d: float, n: int) -> np.ndarray:
    return np.tile([(1 - d) / 2, d, (1 - d) / 2], (n, 1))
