import numpy as np
import pandas as pd
import pytest
from scipy.optimize import check_grad

from smcal.models import poisson
from smcal.models.poisson import GoalModel, fit, fit_before, negative_log_likelihood


def simulate(model: GoalModel, n_rounds: int, seed: int) -> pd.DataFrame:
    """Every ordered pair plays `n_rounds` times; the first-named team is the host."""
    rng = np.random.default_rng(seed)
    rows = []
    day = pd.Timestamp("2000-01-01")
    for r in range(n_rounds):
        for h in model.teams:
            for a in model.teams:
                if h == a:
                    continue
                m = model.score_matrix(h, a, host=h).ravel()
                k = rng.choice(m.size, p=m)
                hg, ag = divmod(k, poisson.MAX_GOALS + 1)
                rows.append((day + pd.Timedelta(days=r), h, a, hg, ag, h))
    return pd.DataFrame(
        rows, columns=["date", "home_team", "away_team", "home_score", "away_score", "host"]
    )


def truth(n_teams: int = 12, rho: float = -0.12) -> GoalModel:
    rng = np.random.default_rng(0)
    att = rng.normal(0, 0.3, n_teams)
    dfn = rng.normal(0, 0.3, n_teams)
    teams = tuple(f"T{i:02d}" for i in range(n_teams))
    return GoalModel(teams, att - att.mean(), dfn - dfn.mean(), 0.15, 0.3, rho)


def test_dixon_coles_recovers_known_parameters() -> None:
    true = truth()
    data = simulate(true, n_rounds=30, seed=1)  # 3,960 matches
    est = fit(data, np.ones(len(data)), ridge=1e-4, dixon_coles=True)
    # The level of att and def is not identified by the likelihood; compare centred values.
    assert np.abs((est.att - est.att.mean()) - true.att).max() < 0.1
    assert np.abs((est.dfn - est.dfn.mean()) - true.dfn).max() < 0.1
    assert est.gamma == pytest.approx(true.gamma, abs=0.05)
    assert est.rho == pytest.approx(true.rho, abs=0.05)


def test_independent_poisson_fixes_rho_at_zero() -> None:
    data = simulate(truth(rho=0.0), n_rounds=5, seed=2)
    est = fit(data, np.ones(len(data)), ridge=1e-4, dixon_coles=False)
    assert est.rho == 0.0


def test_gradient_matches_finite_differences() -> None:
    data = simulate(truth(n_teams=5), n_rounds=3, seed=3)
    w = np.linspace(0.5, 1.0, len(data))
    f, teams = negative_log_likelihood(data, w, ridge=0.5, dixon_coles=True)
    rng = np.random.default_rng(4)
    theta = rng.normal(0, 0.1, 3 + 2 * len(teams))
    theta[2] = -0.08
    err = check_grad(lambda t: f(t)[0], lambda t: f(t)[1], theta)
    assert err < 1e-4 * max(1.0, np.linalg.norm(f(theta)[1]))


def test_outcome_probabilities_sum_to_one_and_favour_the_stronger_team() -> None:
    m = truth()
    strong = m.teams[int(np.argmax(m.att + m.dfn))]
    weak = m.teams[int(np.argmin(m.att + m.dfn))]
    p = m.outcome_probs(strong, weak, host=None)
    assert p.sum() == pytest.approx(1.0)
    assert p[0] > p[2]
    q = m.outcome_probs(weak, strong, host=None)
    assert q[[2, 1, 0]] == pytest.approx(p)


def test_fit_before_uses_only_earlier_matches() -> None:
    data = simulate(truth(n_teams=4), n_rounds=4, seed=5)
    cutoff = pd.Timestamp("2000-01-03")
    later = data["date"] >= cutoff
    poisoned = data.copy()
    poisoned.loc[later, "home_score"] = 9  # would move every estimate if it leaked
    cfg = poisson.FitConfig(xi=0.0, ridge=1.0, window_years=1)
    a = fit_before(data, cutoff, cfg)
    b = fit_before(poisoned, cutoff, cfg)
    assert np.allclose(a.att, b.att) and a.mu == b.mu
