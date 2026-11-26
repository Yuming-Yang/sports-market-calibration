"""Poisson goal models: independent Poisson and Dixon-Coles (1997).

Goals for each side are Poisson with log-means

    log lambda_home = mu + att[home] - def[away] + gamma * [home is host]
    log lambda_away = mu + att[away] - def[home] + gamma * [away is host]

Dixon and Coles multiply the joint probability of the four low scores (0-0, 1-0, 0-1, 1-1)
by a factor tau that depends on one more parameter, rho. rho = 0 gives independent Poisson.

Fitting maximizes a weighted log-likelihood, with weight exp(-xi * age in days), and an L2
penalty on attack and defence strengths. The penalty does two jobs: it pins down the level
(adding a constant to every att and def leaves the likelihood unchanged), and it shrinks
teams with few matches towards average.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

MAX_GOALS = 10
TAU_FLOOR = 1e-10


@dataclass(frozen=True)
class FitConfig:
    xi: float = 0.001  # time decay per day; half-life = ln 2 / xi
    ridge: float = 1.0  # L2 penalty on att and def
    window_years: float = 8.0
    dixon_coles: bool = True


@dataclass(frozen=True)
class GoalModel:
    teams: tuple[str, ...]
    att: np.ndarray
    dfn: np.ndarray
    mu: float
    gamma: float
    rho: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "_lookup", {t: i for i, t in enumerate(self.teams)})

    def rates(self, home: str, away: str, host: str | None) -> tuple[float, float]:
        lk: dict[str, int] = self._lookup  # type: ignore[attr-defined]
        ah = self.att[lk[home]] if home in lk else 0.0
        dh = self.dfn[lk[home]] if home in lk else 0.0
        aa = self.att[lk[away]] if away in lk else 0.0
        da = self.dfn[lk[away]] if away in lk else 0.0
        lh = np.exp(self.mu + ah - da + self.gamma * (host == home))
        la = np.exp(self.mu + aa - dh + self.gamma * (host == away))
        return float(lh), float(la)

    def score_matrix(self, home: str, away: str, host: str | None) -> np.ndarray:
        lh, la = self.rates(home, away, host)
        g = np.arange(MAX_GOALS + 1)
        m = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
        m[0, 0] *= 1 - lh * la * self.rho
        m[0, 1] *= 1 + lh * self.rho
        m[1, 0] *= 1 + la * self.rho
        m[1, 1] *= 1 - self.rho
        m = np.clip(m, 0, None)
        return m / m.sum()

    def outcome_probs(self, home: str, away: str, host: str | None) -> np.ndarray:
        m = self.score_matrix(home, away, host)
        return np.array([np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()])


def _tau_terms(x, y, lh, la, rho):
    """tau and its derivatives with respect to lh, la and rho, per match."""
    tau = np.ones_like(lh)
    d_lh = np.zeros_like(lh)
    d_la = np.zeros_like(lh)
    d_rho = np.zeros_like(lh)

    m00 = (x == 0) & (y == 0)
    tau[m00] = 1 - lh[m00] * la[m00] * rho
    d_lh[m00] = -la[m00] * rho
    d_la[m00] = -lh[m00] * rho
    d_rho[m00] = -lh[m00] * la[m00]

    m01 = (x == 0) & (y == 1)
    tau[m01] = 1 + lh[m01] * rho
    d_lh[m01] = rho
    d_rho[m01] = lh[m01]

    m10 = (x == 1) & (y == 0)
    tau[m10] = 1 + la[m10] * rho
    d_la[m10] = rho
    d_rho[m10] = la[m10]

    m11 = (x == 1) & (y == 1)
    tau[m11] = 1 - rho
    d_rho[m11] = -1.0
    return tau, d_lh, d_la, d_rho


def negative_log_likelihood(
    matches: pd.DataFrame, weights: np.ndarray, ridge: float, dixon_coles: bool
):
    """Return (objective, teams). objective(theta) gives the penalized negative weighted
    log-likelihood and its gradient; theta = [mu, gamma, rho, att..., def...]."""
    teams = tuple(sorted(set(matches["home_team"]) | set(matches["away_team"])))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    hi = matches["home_team"].map(idx).to_numpy()
    ai = matches["away_team"].map(idx).to_numpy()
    host = matches["host"].to_numpy()
    hh = (host == matches["home_team"].to_numpy()).astype(float)
    ah = (host == matches["away_team"].to_numpy()).astype(float)
    x = matches["home_score"].to_numpy().astype(float)
    y = matches["away_score"].to_numpy().astype(float)
    w = np.asarray(weights, dtype=float)

    def unpack(theta):
        return theta[0], theta[1], theta[2], theta[3 : 3 + n], theta[3 + n :]

    def objective(theta):
        mu, gamma, rho, att, dfn = unpack(theta)
        eh = mu + att[hi] - dfn[ai] + gamma * hh
        ea = mu + att[ai] - dfn[hi] + gamma * ah
        lh, la = np.exp(eh), np.exp(ea)
        ll = x * eh - lh + y * ea - la
        g_eh = x - lh
        g_ea = y - la
        g_rho = 0.0
        if dixon_coles:
            tau, d_lh, d_la, d_rho = _tau_terms(x, y, lh, la, rho)
            safe = np.maximum(tau, TAU_FLOOR)
            ll = ll + np.log(safe)
            ok = tau > TAU_FLOOR
            g_eh = g_eh + np.where(ok, d_lh * lh / safe, 0.0)
            g_ea = g_ea + np.where(ok, d_la * la / safe, 0.0)
            g_rho = float(np.sum(w * np.where(ok, d_rho / safe, 0.0)))
        nll = -float(np.sum(w * ll)) + 0.5 * ridge * float(att @ att + dfn @ dfn)

        weh, wea = w * g_eh, w * g_ea
        g_att = np.bincount(hi, weh, n) + np.bincount(ai, wea, n)
        g_dfn = -np.bincount(ai, weh, n) - np.bincount(hi, wea, n)
        grad = np.concatenate(
            [
                [-weh.sum() - wea.sum(), -(weh @ hh + wea @ ah), -g_rho],
                -g_att + ridge * att,
                -g_dfn + ridge * dfn,
            ]
        )
        return nll, grad

    return objective, teams


def fit(
    matches: pd.DataFrame,
    weights: np.ndarray,
    ridge: float,
    dixon_coles: bool = True,
) -> GoalModel:
    """Weighted, penalized maximum likelihood with an analytic gradient."""
    objective, teams = negative_log_likelihood(matches, weights, ridge, dixon_coles)
    n = len(teams)
    x = matches["home_score"].to_numpy().astype(float)
    y = matches["away_score"].to_numpy().astype(float)
    w = np.asarray(weights, dtype=float)
    theta0 = np.zeros(3 + 2 * n)
    theta0[0] = np.log(max((x @ w + y @ w) / (2 * w.sum()), 1e-3))
    bounds = [(None, None), (None, None), (-0.3, 0.3) if dixon_coles else (0.0, 0.0)]
    bounds += [(None, None)] * (2 * n)
    res = minimize(
        objective,
        theta0,
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 5000, "gtol": 1e-7},
    )
    if not res.success:
        raise RuntimeError(f"goal model did not converge: {res.message}")
    mu, gamma, rho = res.x[:3]
    att, dfn = res.x[3 : 3 + n], res.x[3 + n :]
    return GoalModel(teams, att.copy(), dfn.copy(), float(mu), float(gamma), float(rho))


def fit_before(results: pd.DataFrame, date: pd.Timestamp, cfg: FitConfig) -> GoalModel:
    """Fit on matches strictly before `date`, inside the configured window."""
    date = pd.Timestamp(date)
    lo = date - pd.Timedelta(days=round(365.25 * cfg.window_years))
    train = results[(results["date"] < date) & (results["date"] >= lo)]
    if train.empty:
        raise ValueError(f"no matches before {date:%F}")
    age = (date - train["date"]).dt.days.to_numpy()
    return fit(train, np.exp(-cfg.xi * age), cfg.ridge, cfg.dixon_coles)
