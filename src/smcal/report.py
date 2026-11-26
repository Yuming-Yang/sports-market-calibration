"""Build every result table from data/forecasts.csv. README numbers come from here."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from smcal.evaluate import (
    N_BOOT,
    bootstrap_indices,
    brier_decomposition,
    exponent_ci,
    mean_ci,
    paired_difference,
    reliability_table,
)
from smcal.scoring import OUTCOME_INDEX, SCORES

FORECASTERS = {
    "market": ("p_home", "p_draw", "p_away"),
    "dixon_coles": ("dc_home", "dc_draw", "dc_away"),
    "poisson": ("poisson_home", "poisson_draw", "poisson_away"),
    "elo": ("elo_home", "elo_draw", "elo_away"),
    "baseline": ("base_home", "base_draw", "base_away"),
}
PAIRS = [
    ("market", "dixon_coles"),
    ("market", "poisson"),
    ("market", "elo"),
    ("dixon_coles", "poisson"),
    ("dixon_coles", "elo"),
    ("elo", "baseline"),
]
SEED = 20260719
GROUP_STAGE_MATCHES = 72


def probs(fc: pd.DataFrame, name: str) -> np.ndarray:
    return fc[list(FORECASTERS[name])].to_numpy(dtype=float)


def subsets(fc: pd.DataFrame) -> dict[str, pd.DataFrame]:
    fc = fc.sort_values("kickoff", kind="stable").reset_index(drop=True)
    return {
        "all": fc,
        "group": fc.iloc[:GROUP_STAGE_MATCHES].reset_index(drop=True),
        "knockout": fc.iloc[GROUP_STAGE_MATCHES:].reset_index(drop=True),
    }


def build(fc: pd.DataFrame, out: Path, n_boot: int = N_BOOT) -> dict[str, pd.DataFrame]:
    out.mkdir(parents=True, exist_ok=True)
    tables: dict[str, list[dict]] = {
        k: [] for k in ("scores", "pairs", "decomp", "exponent", "outcome_bias")
    }
    for subset, df in subsets(fc).items():
        y = df["outcome"].map(OUTCOME_INDEX).to_numpy()
        idx = bootstrap_indices(len(df), n_boot, SEED)
        each = {
            (name, metric): f(probs(df, name), y)
            for name in FORECASTERS
            for metric, f in SCORES.items()
        }
        for (name, metric), s in each.items():
            m, lo, hi = mean_ci(s, idx)
            tables["scores"].append(
                {
                    "subset": subset,
                    "n": len(df),
                    "forecaster": name,
                    "metric": metric,
                    "mean": m,
                    "ci_lo": lo,
                    "ci_hi": hi,
                }
            )
        for a, b in PAIRS:
            for metric in SCORES:
                d = paired_difference(each[(a, metric)], each[(b, metric)], idx)
                tables["pairs"].append(
                    {
                        "subset": subset,
                        "n": len(df),
                        "a": a,
                        "b": b,
                        "metric": metric,
                        "diff": d.diff,
                        "ci_lo": d.lo,
                        "ci_hi": d.hi,
                        "share_a_better": d.share_a_better,
                    }
                )
        if subset == "all":
            for name in FORECASTERS:
                parts = brier_decomposition(probs(df, name), y, n_bins=10)
                tables["decomp"].append(
                    {
                        "forecaster": name,
                        "reliability": parts.reliability,
                        "resolution": parts.resolution,
                        "uncertainty": parts.uncertainty,
                        "within_bin_variance": parts.within_bin_variance,
                        "within_bin_covariance": parts.within_bin_covariance,
                        "brier": parts.total,
                    }
                )
            # Mean forecast minus observed frequency, per outcome: is one outcome mispriced?
            oh = np.eye(3)[y]
            for name in FORECASTERS:
                for j, label in enumerate(("home", "draw", "away")):
                    gap = probs(df, name)[:, j] - oh[:, j]
                    m, lo, hi = mean_ci(gap, idx)
                    tables["outcome_bias"].append(
                        {
                            "forecaster": name,
                            "outcome": label,
                            "mean_forecast": float(probs(df, name)[:, j].mean()),
                            "observed": float(oh[:, j].mean()),
                            "gap": m,
                            "ci_lo": lo,
                            "ci_hi": hi,
                        }
                    )
            # The exponent refits per resample; 2,000 resamples keep this under a minute.
            small = idx[:2000]
            for name in FORECASTERS:
                if name == "baseline":
                    continue  # constant forecast: alpha is not identified
                a, lo, hi = exponent_ci(probs(df, name), y, small)
                tables["exponent"].append(
                    {"forecaster": name, "alpha": a, "ci_lo": lo, "ci_hi": hi, "n": len(df)}
                )

    frames = {k: pd.DataFrame(v) for k, v in tables.items()}
    rel = []
    all_df = subsets(fc)["all"]
    y_all = all_df["outcome"].map(OUTCOME_INDEX).to_numpy()
    for name in ("market", "dixon_coles", "elo"):
        for row in reliability_table(probs(all_df, name), y_all, n_bins=5):
            rel.append({"forecaster": name, **row})
    frames["reliability"] = pd.DataFrame(rel)

    outcomes = all_df["outcome"].value_counts().reindex(["H", "D", "A"]).fillna(0)
    frames["sample"] = pd.DataFrame(
        {
            "matches": [len(all_df)],
            "home_wins": [int(outcomes["H"])],
            "draws_90min": [int(outcomes["D"])],
            "away_wins": [int(outcomes["A"])],
            "extra_time_inferred": [int(all_df["extra_time"].sum())],
            "raw_sum_min": [float(all_df["raw_sum"].min())],
            "raw_sum_max": [float(all_df["raw_sum"].max())],
            "mean_market_draw_prob": [float(all_df["p_draw"].mean())],
        }
    )
    for name, frame in frames.items():
        frame.to_csv(out / f"{name}.csv", index=False, float_format="%.4f")
    return frames


def figure(fc: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_df = subsets(fc)["all"]
    y = all_df["outcome"].map(OUTCOME_INDEX).to_numpy()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], color="0.7", lw=1, label="perfect calibration")
    styles = {"market": ("C0", "o"), "dixon_coles": ("C1", "s"), "elo": ("C2", "^")}
    for i, (name, (color, marker)) in enumerate(styles.items()):
        rows = [r for r in reliability_table(probs(all_df, name), y, n_bins=5) if r["n"]]
        x = np.array([r["mean_forecast"] for r in rows]) + (i - 1) * 0.006
        obs = np.array([r["observed"] for r in rows])
        err = np.array([[o - r["ci_lo"], r["ci_hi"] - o] for o, r in zip(obs, rows, strict=True)]).T
        ax.errorbar(x, obs, yerr=err, color=color, marker=marker, lw=1, capsize=2, label=name)
    ax.set_xlabel("forecast probability (bin mean)")
    ax.set_ylabel("observed frequency")
    ax.set_title(f"Reliability, {len(all_df)} matches x 3 outcomes")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
