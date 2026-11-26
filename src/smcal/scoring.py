"""Proper scoring rules for three-outcome forecasts.

Forecasts are arrays of shape (n, 3) in the order home win, draw, away win. Outcomes are
integer codes 0, 1, 2 in the same order. Every score here is negatively oriented: lower
is better.
"""

from __future__ import annotations

import numpy as np

OUTCOME_INDEX = {"H": 0, "D": 1, "A": 2}
EPS = 1e-15


def _check(p: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)
    if p.ndim != 2 or p.shape[1] != 3 or p.shape[0] != y.shape[0]:
        raise ValueError("forecasts must be (n, 3) and match the outcomes")
    if not np.allclose(p.sum(axis=1), 1, atol=1e-6) or (p < 0).any():
        raise ValueError("each forecast must be a probability distribution")
    if not np.isin(y, [0, 1, 2]).all():
        raise ValueError("outcomes must be 0, 1 or 2")
    return p, y


def one_hot(y: np.ndarray) -> np.ndarray:
    return np.eye(3)[np.asarray(y, dtype=int)]


def brier_each(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Multiclass Brier score per match: sum over outcomes of (p - o)^2. Range [0, 2]."""
    p, y = _check(p, y)
    return ((p - one_hot(y)) ** 2).sum(axis=1)


def log_loss_each(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p, y = _check(p, y)
    return -np.log(np.clip(p[np.arange(len(y)), y], EPS, None))


def rps_each(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Ranked probability score per match. Respects the order home win < draw < away win:
    putting mass on a draw is penalized less than on the wrong winner. Range [0, 1]."""
    p, y = _check(p, y)
    cp = np.cumsum(p, axis=1)[:, :2]
    co = np.cumsum(one_hot(y), axis=1)[:, :2]
    return ((cp - co) ** 2).sum(axis=1) / 2


def brier(p, y) -> float:
    return float(brier_each(p, y).mean())


def log_loss(p, y) -> float:
    return float(log_loss_each(p, y).mean())


def rps(p, y) -> float:
    return float(rps_each(p, y).mean())


SCORES = {"brier": brier_each, "log_loss": log_loss_each, "rps": rps_each}
