"""Elo ratings for national teams, and win/draw/loss probabilities from rating differences.

The rating update follows the World Football Elo scheme (eloratings.net): a K-factor by
match type, a goal-difference multiplier and a home-advantage term. Ratings give a single
number per match, the rating difference. Three outcome probabilities come from a symmetric
ordered logit on that difference, fitted by maximum likelihood.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

MAJOR_FINALS = {
    "UEFA Euro",
    "Copa América",
    "African Cup of Nations",
    "AFC Asian Cup",
    "Gold Cup",
    "Confederations Cup",
    "Oceania Nations Cup",
}


def k_factor(tournament: str) -> float:
    """K by match type, from the World Football Elo scheme."""
    if tournament == "FIFA World Cup":
        return 60.0
    if tournament in MAJOR_FINALS:
        return 50.0
    if "qualification" in tournament or "Nations League" in tournament:
        return 40.0
    if tournament == "Friendly":
        return 20.0
    return 30.0


def goal_multiplier(goal_diff: int) -> float:
    n = abs(goal_diff)
    if n <= 1:
        return 1.0
    if n == 2:
        return 1.5
    return 1.75 + (n - 3) / 8


@dataclass(frozen=True)
class EloParams:
    k_scale: float = 1.0  # multiplies the K table; tuned on pre-2026 data
    home_adv: float = 100.0  # rating points added to the host
    initial: float = 1500.0


def home_edge(home: str, away: str, host: str | None, home_adv: float) -> float:
    if host == home:
        return home_adv
    if host == away:
        return -home_adv
    return 0.0


def _expected(dr: float) -> float:
    return 1.0 / (10 ** (-dr / 400) + 1)


def run_elo(matches: pd.DataFrame, params: EloParams) -> tuple[np.ndarray, dict[str, float]]:
    """Process matches in date order.

    Returns each match's pre-match rating difference (home minus away, including home
    advantage), aligned with the rows of `matches`, and the final ratings. Ratings are
    frozen within a day: every match on date D is rated with ratings built from matches
    before D, so no match is ever rated with a result from the same day or later.
    """
    df = matches.reset_index(drop=True)
    if not df["date"].is_monotonic_increasing:
        raise ValueError("matches must be sorted by date")
    ratings: dict[str, float] = {}
    out = np.empty(len(df))
    for _, day in df.groupby("date", sort=True):
        updates: dict[str, float] = {}
        for idx, row in zip(day.index, day.itertuples(index=False), strict=True):
            h, a = row.home_team, row.away_team
            rh, ra = ratings.get(h, params.initial), ratings.get(a, params.initial)
            hst = row.host if isinstance(row.host, str) else None
            dr = rh - ra + home_edge(h, a, hst, params.home_adv)
            out[idx] = dr
            gd = int(row.home_score) - int(row.away_score)
            w = 1.0 if gd > 0 else 0.0 if gd < 0 else 0.5
            k = params.k_scale * k_factor(row.tournament) * goal_multiplier(gd)
            delta = k * (w - _expected(dr))
            updates[h] = updates.get(h, 0.0) + delta
            updates[a] = updates.get(a, 0.0) - delta
        for team, d in updates.items():
            ratings[team] = ratings.get(team, params.initial) + d
    return out, ratings


# --- outcome probabilities -------------------------------------------------------------


@dataclass(frozen=True)
class OrderedLogit:
    """P(H) = s(b*x - c), P(A) = s(-b*x - c), P(D) = 1 - P(H) - P(A), with x = dr / 100.

    Symmetric in home and away, which suits neutral venues where the labels are arbitrary;
    home advantage enters through the rating difference itself.
    """

    slope: float
    cut: float

    def probs(self, rating_diff: Iterable[float]) -> np.ndarray:
        x = np.asarray(rating_diff, dtype=float) / 100
        ph = expit(self.slope * x - self.cut)
        pa = expit(-self.slope * x - self.cut)
        return np.column_stack([ph, 1 - ph - pa, pa])


def fit_ordered_logit(rating_diff: np.ndarray, outcome: np.ndarray) -> OrderedLogit:
    """Maximum likelihood. `outcome` holds 0 (home win), 1 (draw), 2 (away win)."""
    x = np.asarray(rating_diff, dtype=float) / 100
    y = np.asarray(outcome)
    if len(y) < 30 or len(np.unique(y)) < 3:
        raise ValueError("too few matches, or not all three outcomes, to fit the link")

    def nll(theta: np.ndarray) -> float:
        b, log_c = theta
        p = OrderedLogit(b, float(np.exp(log_c))).probs(x * 100)
        return -float(np.sum(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None))))

    res = minimize(
        nll,
        x0=np.array([0.5, np.log(0.5)]),
        method="Nelder-Mead",
        options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 5000},
    )
    if not res.success:
        raise RuntimeError(f"ordered logit did not converge: {res.message}")
    return OrderedLogit(float(res.x[0]), float(np.exp(res.x[1])))
