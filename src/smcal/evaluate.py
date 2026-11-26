"""Evaluation: intervals, paired comparisons, calibration and the Brier decomposition.

With about a hundred matches, every number needs an interval. Intervals here come from a
bootstrap over matches: a match is the unit that is sampled, so the three outcome
probabilities of one match always stay together.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm

from smcal.scoring import EPS, one_hot

N_BOOT = 10_000
CI = 0.95


def bootstrap_indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_boot, n))


def mean_ci(values: np.ndarray, idx: np.ndarray) -> tuple[float, float, float]:
    """Mean and percentile bootstrap interval of a per-match quantity."""
    v = np.asarray(values, dtype=float)
    means = v[idx].mean(axis=1)
    a = (1 - CI) / 2
    return float(v.mean()), float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


@dataclass(frozen=True)
class PairedDiff:
    diff: float  # mean score of A minus mean score of B; negative means A is better
    lo: float
    hi: float
    share_a_better: float  # share of bootstrap resamples in which A scores lower


def paired_difference(score_a: np.ndarray, score_b: np.ndarray, idx: np.ndarray) -> PairedDiff:
    """Paired bootstrap: resample matches, keep both forecasters' scores on the same match.

    Pairing matters. Two forecasters' scores on the same match are strongly correlated
    (both suffer on an upset), so the difference is far less noisy than either score.
    """
    d = np.asarray(score_a, dtype=float) - np.asarray(score_b, dtype=float)
    m, lo, hi = mean_ci(d, idx)
    share = float((d[idx].mean(axis=1) < 0).mean())
    return PairedDiff(m, lo, hi, share)


# --- Brier decomposition ---------------------------------------------------------------


@dataclass(frozen=True)
class BrierParts:
    reliability: float
    resolution: float
    uncertainty: float
    within_bin_variance: float
    within_bin_covariance: float

    @property
    def total(self) -> float:
        return (
            self.reliability
            - self.resolution
            + self.uncertainty
            + self.within_bin_variance
            - self.within_bin_covariance
        )


def bin_edges(n_bins: int) -> np.ndarray:
    return np.linspace(0, 1, n_bins + 1)


def _bin(f: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(edges, f, side="right") - 1, 0, len(edges) - 2)


def brier_decomposition_binary(f: np.ndarray, o: np.ndarray, n_bins: int = 10) -> BrierParts:
    """Murphy's decomposition with binned forecasts, made exact.

    Murphy (1973): Brier = reliability - resolution + uncertainty, exact when forecasts take
    a few discrete values. With continuous forecasts placed in bins, two within-bin terms
    complete the identity (Stephenson, Coelho and Jolliffe, 2008):

        Brier = REL - RES + UNC + WBV - WBC
    """
    f = np.asarray(f, dtype=float)
    o = np.asarray(o, dtype=float)
    n = len(f)
    k = _bin(f, bin_edges(n_bins))
    obar = o.mean()
    rel = res = wbv = wbc = 0.0
    for b in np.unique(k):
        m = k == b
        fb, ob = f[m].mean(), o[m].mean()
        rel += m.sum() * (fb - ob) ** 2
        res += m.sum() * (ob - obar) ** 2
        wbv += ((f[m] - fb) ** 2).sum()
        wbc += 2 * ((o[m] - ob) * (f[m] - fb)).sum()
    return BrierParts(rel / n, res / n, obar * (1 - obar), wbv / n, wbc / n)


def brier_decomposition(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> BrierParts:
    """Multiclass Brier is the sum of three one-against-rest Brier scores; so are its parts."""
    oh = one_hot(y)
    parts = [brier_decomposition_binary(p[:, j], oh[:, j], n_bins) for j in range(3)]
    return BrierParts(
        *(float(sum(getattr(x, f) for x in parts)) for f in BrierParts.__dataclass_fields__)
    )


# --- reliability ------------------------------------------------------------------------


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    z = norm.ppf(0.5 + CI / 2)
    phat = k / n
    denom = 1 + z**2 / n
    centre = (phat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return float(centre - half), float(centre + half)


def reliability_table(p: np.ndarray, y: np.ndarray, n_bins: int = 5) -> list[dict]:
    """Pool the three one-against-rest forecasts of every match and bin them.

    The Wilson interval treats the pooled forecasts as independent. They are not quite (the
    three outcomes of one match are linked), so the intervals are somewhat too narrow.
    """
    f = p.ravel()
    o = one_hot(y).ravel()
    k = _bin(f, bin_edges(n_bins))
    edges = bin_edges(n_bins)
    rows = []
    for b in range(n_bins):
        m = k == b
        n = int(m.sum())
        hits = int(o[m].sum())
        lo, hi = wilson(hits, n)
        rows.append(
            {
                "bin_lo": float(edges[b]),
                "bin_hi": float(edges[b + 1]),
                "n": n,
                "mean_forecast": float(f[m].mean()) if n else float("nan"),
                "observed": hits / n if n else float("nan"),
                "ci_lo": lo,
                "ci_hi": hi,
            }
        )
    return rows


# --- favourite-longshot check -------------------------------------------------------------


def calibration_exponent(p: np.ndarray, y: np.ndarray) -> float:
    """The alpha that maximizes the likelihood of q proportional to p ** alpha.

    alpha = 1: no recalibration improves the forecast this way. alpha > 1: the forecast
    should be sharper, i.e. favourites are underpriced and longshots overpriced, which is
    the favourite-longshot bias. alpha < 1: the forecast is overconfident.
    """
    logp = np.log(np.clip(p, EPS, None))
    rows = np.arange(len(y))

    def nll(alpha: float) -> float:
        z = alpha * logp
        z = z - z.max(axis=1, keepdims=True)
        return -float((z[rows, y] - np.log(np.exp(z).sum(axis=1))).sum())

    res = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded", options={"xatol": 1e-8})
    return float(res.x)


def exponent_ci(p: np.ndarray, y: np.ndarray, idx: np.ndarray) -> tuple[float, float, float]:
    point = calibration_exponent(p, y)
    boots = np.array([calibration_exponent(p[i], y[i]) for i in idx])
    a = (1 - CI) / 2
    return point, float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a))
