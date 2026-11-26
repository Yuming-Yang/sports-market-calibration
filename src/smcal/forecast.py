"""Hyperparameter tuning on pre-2026 data, and forecasts for the World Cup matches.

Tuning uses only matches before 2026-01-01. Its output is frozen in `config/frozen.json`
before any World Cup forecast is scored.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict

import numpy as np
import pandas as pd

from smcal.models.baseline import baseline_probs, draw_rate
from smcal.models.elo import EloParams, fit_ordered_logit, run_elo
from smcal.models.poisson import FitConfig, fit_before
from smcal.scoring import OUTCOME_INDEX, log_loss

TUNE_FIT = ("2010-01-01", "2023-01-01")  # ordered-logit fit for Elo tuning
TUNE_VALID = ("2023-01-01", "2026-01-01")  # validation matches
FINAL_FIT = ("2010-01-01", "2026-01-01")  # ordered-logit fit for the World Cup forecasts

ELO_GRID = {"k_scale": [0.5, 0.75, 1.0, 1.5], "home_adv": [50.0, 100.0, 150.0]}
GOAL_GRID = {"xi": [0.0, 0.0005, 0.001, 0.002], "ridge": [0.1, 0.3, 1.0, 3.0]}


def competitive(df: pd.DataFrame) -> pd.Series:
    return df["tournament"] != "Friendly"


def between(df: pd.DataFrame, lo: str, hi: str) -> pd.Series:
    return (df["date"] >= pd.Timestamp(lo)) & (df["date"] < pd.Timestamp(hi))


def outcome_codes(df: pd.DataFrame) -> np.ndarray:
    gd = df["home_score"].to_numpy() - df["away_score"].to_numpy()
    return np.where(gd > 0, 0, np.where(gd == 0, 1, 2))


def tune_elo(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    y = outcome_codes(results)
    fit_mask = (between(results, *TUNE_FIT) & competitive(results)).to_numpy()
    val_mask = (between(results, *TUNE_VALID) & competitive(results)).to_numpy()
    for k, h in itertools.product(ELO_GRID["k_scale"], ELO_GRID["home_adv"]):
        params = EloParams(k_scale=k, home_adv=h)
        diffs, _ = run_elo(results, params)
        link = fit_ordered_logit(diffs[fit_mask], y[fit_mask])
        p = link.probs(diffs[val_mask])
        rows.append({"k_scale": k, "home_adv": h, "log_loss": log_loss(p, y[val_mask])})
    return pd.DataFrame(rows).sort_values("log_loss", kind="stable").reset_index(drop=True)


def quarter_starts(lo: str, hi: str) -> list[pd.Timestamp]:
    return list(pd.date_range(lo, hi, freq="QS", inclusive="left"))


def tune_goals(results: pd.DataFrame) -> pd.DataFrame:
    """Refit at the start of each validation quarter; score that quarter's matches."""
    origins = quarter_starts(*TUNE_VALID)
    rows = []
    for xi, ridge in itertools.product(GOAL_GRID["xi"], GOAL_GRID["ridge"]):
        cfg = FitConfig(xi=xi, ridge=ridge, dixon_coles=False)
        probs, ys = [], []
        for start in origins:
            end = start + pd.DateOffset(months=3)
            test = results[between(results, f"{start:%F}", f"{end:%F}") & competitive(results)]
            if test.empty:
                continue
            model = fit_before(results, start, cfg)
            probs.append(_goal_probs(model, test))
            ys.append(outcome_codes(test))
        p, y = np.vstack(probs), np.concatenate(ys)
        rows.append({"xi": xi, "ridge": ridge, "log_loss": log_loss(p, y)})
    return pd.DataFrame(rows).sort_values("log_loss", kind="stable").reset_index(drop=True)


def _host(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _goal_probs(model, matches: pd.DataFrame) -> np.ndarray:
    return np.array(
        [
            model.outcome_probs(r.home_team, r.away_team, _host(r.host))
            for r in matches.itertuples(index=False)
        ]
    )


def world_cup_forecasts(
    results: pd.DataFrame, table: pd.DataFrame, elo: EloParams, goals: FitConfig
) -> pd.DataFrame:
    """Model probabilities for each match in `table`, using only earlier matches.

    `table` holds the World Cup matches (slug, kickoff, home, away, host), oriented as on
    Polymarket. `results` must contain them too, since Elo is run through the tournament.
    """
    out = table[["slug", "kickoff", "home", "away", "host"]].copy()
    kick = pd.to_datetime(out["kickoff"], utc=True)
    # The results file uses local dates; a match's local date is its UTC date or the day
    # before. The model cut-off is the earlier of the two, so it never includes the match.
    cutoff = (kick.dt.tz_localize(None).dt.normalize() - pd.Timedelta(days=1)).where(
        kick.dt.hour < 12, kick.dt.tz_localize(None).dt.normalize()
    )
    out["model_cutoff"] = cutoff

    # Elo: ratings as of each match's local date, from one pass through all results.
    y = outcome_codes(results)
    diffs, _ = run_elo(results, elo)
    fit_mask = (between(results, *FINAL_FIT) & competitive(results)).to_numpy()
    link = fit_ordered_logit(diffs[fit_mask], y[fit_mask])
    elo_p = []
    for r in out.itertuples(index=False):
        hit = results.index[
            (results["home_team"].isin([r.home, r.away]))
            & (results["away_team"].isin([r.home, r.away]))
            & ((results["date"] - r.model_cutoff).abs() <= pd.Timedelta(days=1))
            & (results["tournament"] == "FIFA World Cup")
        ]
        if len(hit) != 1:
            raise LookupError(f"{len(hit)} result rows for {r.slug}")
        i = results.index.get_loc(hit[0])
        dr = diffs[i] if results.iloc[i]["home_team"] == r.home else -diffs[i]
        elo_p.append(link.probs([dr])[0])
    out[["elo_home", "elo_draw", "elo_away"]] = np.array(elo_p)

    # Goal models: refit once per cut-off date.
    for name, dc in (("poisson", False), ("dc", True)):
        cfg = FitConfig(**{**asdict(goals), "dixon_coles": dc})
        probs = np.empty((len(out), 3))
        for cut, idx in out.groupby("model_cutoff").groups.items():
            model = fit_before(results, cut, cfg)
            for i in idx:
                r = out.loc[i]
                probs[out.index.get_loc(i)] = model.outcome_probs(r.home, r.away, _host(r.host))
        out[[f"{name}_home", f"{name}_draw", f"{name}_away"]] = probs

    d = draw_rate(results, out["model_cutoff"].min())
    out[["base_home", "base_draw", "base_away"]] = baseline_probs(d, len(out))
    return out


__all__ = ["OUTCOME_INDEX", "tune_elo", "tune_goals", "world_cup_forecasts"]
