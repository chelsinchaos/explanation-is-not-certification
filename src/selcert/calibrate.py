"""Certified selective deployment (Theorem 5).

Given calibration scores s_j and 0-1 losses l_j drawn i.i.d. from the deployment
distribution, with the score, the model and the grid all fixed independently of
the calibration sample, `calibrate_bonferroni` returns a threshold tau_hat such
that, with probability at least 1 - delta over the calibration draw, the
selective risk  R(tau_hat) = P(error | s <= tau_hat)  is at most alpha.

Two procedures are provided.

* Bonferroni (the paper's Theorem 5): a one-sided Clopper-Pearson upper bound
  at level 1 - delta/K at each of K grid points, tau_hat = the largest grid
  point whose bound is at most alpha.  Valid for any grid fixed in advance and
  any shape of the risk curve.
* Fixed-sequence (Learn then Test, Angelopoulos et al. 2021): test grid points in
  increasing order, each at level 1 - delta, stop at the first failure.  Also
  valid for any fixed grid; tighter when the risk is monotone in tau.

The machinery is classical (Geifman & El-Yaniv 2017; Bates et al. 2021;
Angelopoulos et al. 2021); it is reproduced here so that the paper's claims can
be checked line by line.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.stats import beta as beta_dist


def clopper_pearson_upper(errors, trials, confidence: float):
    """One-sided Clopper-Pearson upper confidence bound for a binomial proportion.

    Returns sup{ p : P_{Bin(trials, p)}[S <= errors] >= 1 - confidence }, which
    equals the (confidence)-quantile of Beta(errors + 1, trials - errors); it is
    1 when errors == trials, and 1 when trials == 0.  Vectorised over arrays.
    """
    e = np.asarray(errors, dtype=float)
    m = np.asarray(trials, dtype=float)
    out = np.ones(np.broadcast(e, m).shape, dtype=float)
    ok = (m > 0) & (e < m)
    e_ok = np.broadcast_to(e, out.shape)[ok]
    m_ok = np.broadcast_to(m, out.shape)[ok]
    out[ok] = beta_dist.ppf(confidence, e_ok + 1.0, m_ok - e_ok)
    return out


@dataclass
class CalibrationResult:
    tau_hat: Optional[float]          # None means: defer everything
    grid: np.ndarray
    accepted: np.ndarray              # m_k per grid point
    errors: np.ndarray                # e_k per grid point
    upper_bounds: np.ndarray          # UCB_k per grid point
    alpha: float
    delta: float
    method: str

    @property
    def empirical_coverage(self) -> float:
        """Fraction of the calibration sample that would be auto-accepted."""
        if self.tau_hat is None:
            return 0.0
        k = int(np.searchsorted(self.grid, self.tau_hat, side="right") - 1)
        return float(self.accepted[k] / max(self._n, 1))

    _n: int = 0


def _tabulate(scores, losses, grid):
    s = np.asarray(scores, dtype=float)
    l = np.asarray(losses, dtype=float)
    if s.shape != l.shape or s.ndim != 1:
        raise ValueError("scores and losses must be 1-d arrays of equal length")
    if not np.all((l == 0) | (l == 1)):
        raise ValueError("losses must be 0/1")
    g = np.asarray(grid, dtype=float)
    if np.any(np.diff(g) <= 0):
        raise ValueError("grid must be strictly increasing")
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    cum_err = np.concatenate([[0.0], np.cumsum(l[order])])
    m_k = np.searchsorted(s_sorted, g, side="right")      # number of s_j <= tau_k
    e_k = cum_err[m_k]
    return s.size, m_k.astype(float), e_k


def calibrate_bonferroni(scores, losses, grid: Sequence[float], alpha: float,
                         delta: float) -> CalibrationResult:
    """Theorem 5: Clopper-Pearson at level 1 - delta/K on each of K fixed grid points."""
    _check_levels(alpha, delta)
    n, m_k, e_k = _tabulate(scores, losses, grid)
    K = len(grid)
    ucb = clopper_pearson_upper(e_k, m_k, 1.0 - delta / K)
    passing = np.flatnonzero(ucb <= alpha)
    tau_hat = float(np.asarray(grid)[passing.max()]) if passing.size else None
    res = CalibrationResult(tau_hat, np.asarray(grid, float), m_k, e_k, ucb,
                            alpha, delta, "bonferroni")
    res._n = n
    return res


def calibrate_fixed_sequence(scores, losses, grid: Sequence[float], alpha: float,
                             delta: float) -> CalibrationResult:
    """Learn-then-Test fixed-sequence variant: test tau_1 < tau_2 < ... each at level
    1 - delta; stop at the first grid point whose bound exceeds alpha."""
    _check_levels(alpha, delta)
    n, m_k, e_k = _tabulate(scores, losses, grid)
    ucb = clopper_pearson_upper(e_k, m_k, 1.0 - delta)
    tau_hat = None
    for k, u in enumerate(ucb):
        if u <= alpha:
            tau_hat = float(grid[k])
        else:
            break
    res = CalibrationResult(tau_hat, np.asarray(grid, float), m_k, e_k, ucb,
                            alpha, delta, "fixed-sequence")
    res._n = n
    return res


def _check_levels(alpha: float, delta: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
