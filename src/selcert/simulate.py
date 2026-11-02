"""Simulation studies reported in the paper (Methods, "Simulation study").

Study A  validity of the certified threshold (Theorem 5) on a synthetic
         deployment distribution with a known risk curve, for the valid
         procedure and for two invalid variants that drop one hypothesis each.
Study B  the per-threshold confidence statement fails when the threshold is an
         order statistic of the calibration scores (the hypothesis "grid fixed
         independently of the sample" is necessary).
Study C  certified coverage as a function of score quality (AUROC) at fixed
         alpha, delta and calibration size: the guarantee is unchanged, only
         coverage moves (Proposition 2).

All studies use closed-form or large-sample ground truth so that the true
selective risk of every chosen threshold is known.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List

import math

import numpy as np
from scipy.stats import multivariate_normal, norm

from .calibrate import clopper_pearson_upper


# --------------------------------------------------------------------------- #
# Study A: validity on a synthetic deployment distribution
# --------------------------------------------------------------------------- #

def risk_curve_A(tau: np.ndarray, q0: float = 0.0025, q2: float = 0.6) -> np.ndarray:
    """R(tau) = P(error | s <= tau) for s ~ U(0,1) and P(error | s) = q0 + q2 s^2.

    With the defaults R(tau) = 0.0025 + 0.2 tau^2, so that no grid point k/50 sits exactly on
    R = 0.01 or R = 0.05 (oracle thresholds 0.1936 and 0.4873)."""
    return q0 + q2 * tau ** 2 / 3.0


def _tabulate_rows(s_sorted: np.ndarray, err_sorted: np.ndarray, grid: np.ndarray):
    """m_k and e_k for every replication (rows) and grid point (columns)."""
    R, _ = s_sorted.shape
    cum = np.concatenate([np.zeros((R, 1)), np.cumsum(err_sorted, axis=1)], axis=1)
    m_k = np.empty((R, grid.size), dtype=np.int64)
    for r in range(R):
        m_k[r] = np.searchsorted(s_sorted[r], grid, side="right")
    e_k = np.take_along_axis(cum, m_k, axis=1)
    return m_k.astype(float), e_k


def _select_max_passing(ucb: np.ndarray, grid: np.ndarray, alpha: float):
    """tau_hat = largest grid point with UCB <= alpha; NaN means defer everything."""
    passing = ucb <= alpha
    any_pass = passing.any(axis=1)
    last = grid.size - 1 - np.argmax(passing[:, ::-1], axis=1)
    tau_hat = np.where(any_pass, grid[last], np.nan)
    return tau_hat


def _select_fixed_sequence(ucb: np.ndarray, grid: np.ndarray, alpha: float):
    passing = ucb <= alpha
    # first failure index per row; prefix of passes
    fail = ~passing
    first_fail = np.where(fail.any(axis=1), np.argmax(fail, axis=1), grid.size)
    tau_hat = np.where(first_fail > 0, grid[np.clip(first_fail - 1, 0, grid.size - 1)], np.nan)
    return tau_hat


@dataclass
class StudyARow:
    variant: str
    m: int
    alpha: float
    delta: float
    reps: int
    violation_rate: float
    defer_all_rate: float
    mean_coverage: float
    oracle_coverage: float


def study_A(m_values=(250, 500, 1000, 2000, 4000, 8000), alphas=(0.01, 0.05),
            delta: float = 0.05, K: int = 50, reps: int = 10000, seed: int = 0,
            q0: float = 0.0025, q2: float = 0.6) -> List[StudyARow]:
    rng = np.random.default_rng(seed)
    grid = np.arange(1, K + 1) / K
    rows: List[StudyARow] = []
    for m in m_values:
        s = rng.uniform(size=(reps, m))
        err = (rng.uniform(size=(reps, m)) < (q0 + q2 * s ** 2)).astype(float)
        order = np.argsort(s, axis=1)
        s_sorted = np.take_along_axis(s, order, axis=1)
        err_sorted = np.take_along_axis(err, order, axis=1)
        m_k, e_k = _tabulate_rows(s_sorted, err_sorted, grid)
        ucb_bonf = clopper_pearson_upper(e_k, m_k, 1.0 - delta / K)
        ucb_uncorr = clopper_pearson_upper(e_k, m_k, 1.0 - delta)
        # sample-quantile grid (data dependent) with Bonferroni
        q_idx = np.clip(np.round(grid * m).astype(int) - 1, 0, m - 1)
        tau_q = s_sorted[:, q_idx]                       # (reps, K) data-dependent grid
        m_q = (q_idx + 1).astype(float)[None, :].repeat(reps, 0)
        cum = np.concatenate([np.zeros((reps, 1)), np.cumsum(err_sorted, axis=1)], axis=1)
        e_q = np.take_along_axis(cum, (q_idx + 1)[None, :].repeat(reps, 0), axis=1)
        ucb_q = clopper_pearson_upper(e_q, m_q, 1.0 - delta / K)
        for alpha in alphas:
            true_curve = risk_curve_A(grid, q0, q2)
            feasible = grid[true_curve <= alpha + 1e-12]
            oracle_cov = float(feasible.max()) if feasible.size else 0.0
            variants = {
                "bonferroni_fixed_grid": _select_max_passing(ucb_bonf, grid, alpha),
                "fixed_sequence_fixed_grid": _select_fixed_sequence(ucb_uncorr, grid, alpha),
                "uncorrected_fixed_grid": _select_max_passing(ucb_uncorr, grid, alpha),
            }
            # data-dependent grid: tau_hat is the largest sample quantile passing
            passing_q = ucb_q <= alpha
            any_pass = passing_q.any(axis=1)
            last = K - 1 - np.argmax(passing_q[:, ::-1], axis=1)
            tau_hat_q = np.where(any_pass, np.take_along_axis(tau_q, last[:, None], axis=1)[:, 0], np.nan)
            variants["bonferroni_sample_quantile_grid"] = tau_hat_q
            for name, tau_hat in variants.items():
                deferred = np.isnan(tau_hat)
                true_risk = np.where(deferred, 0.0, risk_curve_A(np.nan_to_num(tau_hat), q0, q2))
                violation = float(np.mean((~deferred) & (true_risk > alpha)))
                cov = float(np.mean(np.where(deferred, 0.0, np.nan_to_num(tau_hat))))
                rows.append(StudyARow(name, m, alpha, delta, reps, violation,
                                      float(deferred.mean()), cov, oracle_cov))
    return rows


# --------------------------------------------------------------------------- #
# Study B: per-threshold statement fails for an order-statistic threshold
# --------------------------------------------------------------------------- #

@dataclass
class StudyBRow:
    threshold: str
    m: int
    j: int
    delta: float
    reps: int
    per_threshold_violation: float


def study_B(m: int = 50, j: int = 20, c: float = 0.3, delta: float = 0.05,
            reps: int = 2_000_000, seed: int = 1) -> List[StudyBRow]:
    """s ~ U(0,1), P(error | s) = 1[s < c].  Threshold either the j-th order
    statistic of the calibration scores (data dependent) or the fixed value
    j/(m+1) (its expectation).  Reports P[R(tau) > UCB(tau)] against delta."""
    rng = np.random.default_rng(seed)
    out: List[StudyBRow] = []
    s = rng.uniform(size=(reps, m))
    s.sort(axis=1)
    # data-dependent threshold: the j-th order statistic
    tau = s[:, j - 1]
    e = (s[:, :j] < c).sum(axis=1)
    true_risk = np.minimum(1.0, c / tau)
    ucb = clopper_pearson_upper(e, np.full(reps, j), 1.0 - delta)
    out.append(StudyBRow("j-th order statistic (data dependent)", m, j, delta, reps,
                         float(np.mean(true_risk > ucb))))
    # fixed threshold at the expectation of that order statistic
    tau0 = j / (m + 1)
    acc = s <= tau0
    m0 = acc.sum(axis=1)
    e0 = ((s < c) & acc).sum(axis=1)
    ucb0 = clopper_pearson_upper(e0, m0, 1.0 - delta)
    r0 = min(1.0, c / tau0)
    out.append(StudyBRow("fixed threshold j/(m+1)", m, j, delta, reps,
                         float(np.mean(r0 > ucb0))))
    return out


# --------------------------------------------------------------------------- #
# Study C: certified coverage as a function of score quality
# --------------------------------------------------------------------------- #

@dataclass
class StudyCRow:
    sigma: float
    auroc: float
    m: int
    alpha: float
    delta: float
    reps: int
    violation_rate: float
    defer_all_rate: float
    mean_certified_coverage: float
    oracle_coverage: float


def study_C(sigmas=(0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0), base_rate: float = 0.10,
            m: int = 2000, alpha: float = 0.05, delta: float = 0.05, K: int = 60,
            reps: int = 2000, truth_samples: int = 4_000_000, seed: int = 2) -> List[StudyCRow]:
    """Latent z ~ N(0,1); the model errs when z > z0 with P(z > z0) = base_rate.
    Score s = z + sigma * noise.  Larger sigma means a worse score (lower AUROC).
    The grid is fixed in advance: K points on [-3, 3].  Coverage and selective risk
    on the grid are computed in closed form from the bivariate normal distribution
    of (z, s); only the AUROC label is estimated by Monte Carlo."""
    rng = np.random.default_rng(seed)
    z0 = norm.ppf(1.0 - base_rate)
    grid = np.linspace(-3.0, 3.0, K)
    rows: List[StudyCRow] = []
    for sigma in sigmas:
        # ground truth on the grid in closed form: (z, s) is bivariate normal with
        # Var(s) = 1 + sigma^2 and Corr(z, s) = 1 / sqrt(1 + sigma^2).
        sd_s = math.sqrt(1.0 + sigma ** 2)
        rho = 1.0 / sd_s
        C_true = norm.cdf(grid / sd_s)
        cov = [[1.0, rho], [rho, 1.0]]
        joint_ok = np.array([multivariate_normal.cdf([z0, g / sd_s], mean=[0.0, 0.0], cov=cov) for g in grid])
        R_true = np.where(C_true > 0, (C_true - joint_ok) / np.maximum(C_true, 1e-300), 0.0)
        # AUROC (a descriptive label for the score) by Monte Carlo
        z = rng.standard_normal(truth_samples)
        s = z + sigma * rng.standard_normal(truth_samples)
        err = z > z0
        auroc = _auroc(s[err], s[~err], rng)
        feasible = C_true[R_true <= alpha]
        oracle = float(feasible.max()) if feasible.size else 0.0
        # calibration replications
        zc = rng.standard_normal((reps, m))
        sc = zc + sigma * rng.standard_normal((reps, m))
        ec = (zc > z0).astype(float)
        order_c = np.argsort(sc, axis=1)
        sc_sorted = np.take_along_axis(sc, order_c, axis=1)
        ec_sorted = np.take_along_axis(ec, order_c, axis=1)
        m_k, e_k = _tabulate_rows(sc_sorted, ec_sorted, grid)
        ucb = clopper_pearson_upper(e_k, m_k, 1.0 - delta / K)
        passing = ucb <= alpha
        any_pass = passing.any(axis=1)
        last = K - 1 - np.argmax(passing[:, ::-1], axis=1)
        risk_hat = np.where(any_pass, R_true[last], 0.0)
        cov_hat = np.where(any_pass, C_true[last], 0.0)
        rows.append(StudyCRow(float(sigma), float(auroc), m, alpha, delta, reps,
                              float(np.mean(any_pass & (risk_hat > alpha))),
                              float(np.mean(~any_pass)), float(cov_hat.mean()), oracle))
    return rows


def _auroc(pos: np.ndarray, neg: np.ndarray, rng: np.random.Generator, n: int = 400_000) -> float:
    p = rng.choice(pos, size=n)
    q = rng.choice(neg, size=n)
    return float(np.mean(p > q) + 0.5 * np.mean(p == q))


def rows_to_dicts(rows) -> List[Dict]:
    return [asdict(r) for r in rows]
