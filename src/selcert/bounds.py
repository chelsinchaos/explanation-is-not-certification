"""Counting bounds on explainable behaviour tables (Theorem 2).

For n distinct audit inputs, |Y| labels, a budget of ell bits and a 0-1
residual tolerance eps < 1/2, the fraction of behaviour tables in Y^n that admit
an eps-faithful explanation of length <= ell is at most

    2 ** ( ell - n * B(eps, |Y|) ),
    B(eps, q) = log2 q - h(eps) - eps log2 (q - 1),

where h is the binary entropy function.  B is the worst-case explanation budget
per audited output.  B(eps, q) > 0 for every q >= 2 and eps < 1/2.
"""
from __future__ import annotations

import math
from functools import lru_cache


def binary_entropy(p: float) -> float:
    """h(p) in bits, with h(0) = h(1) = 0."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def budget_per_output(eps: float, n_outputs: int) -> float:
    """B(eps, |Y|) = log2|Y| - h(eps) - eps log2(|Y| - 1)."""
    if n_outputs < 2:
        raise ValueError("|Y| must be at least 2")
    if not 0.0 <= eps <= 1.0:
        raise ValueError("eps must lie in [0, 1]")
    tail = eps * math.log2(n_outputs - 1) if n_outputs > 2 else 0.0
    return math.log2(n_outputs) - binary_entropy(eps) - tail


@lru_cache(maxsize=None)
def _log2_binom(n: int, k: int) -> float:
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)) / math.log(2.0)


def hamming_ball_size(n: int, radius: int, n_outputs: int) -> float:
    """log2 of the number of tables within Hamming distance `radius` of a fixed table.

    Exact value of log2 sum_{k<=radius} C(n,k) (|Y|-1)^k, computed in log space.
    """
    if radius < 0:
        return float("-inf")
    log2_q1 = math.log2(n_outputs - 1)          # = 0 when |Y| = 2
    terms = [_log2_binom(n, k) + k * log2_q1 for k in range(0, min(radius, n) + 1)]
    m = max(terms)
    return m + math.log2(sum(2.0 ** (t - m) for t in terms))


def entropy_bound_log2(n: int, eps: float, n_outputs: int) -> float:
    """log2 of the entropy upper bound  2^{n h(eps)} (|Y|-1)^{eps n}  (valid for eps <= 1/2)."""
    if eps > 0.5:
        raise ValueError("entropy bound is only used for eps <= 1/2")
    tail = eps * n * math.log2(n_outputs - 1) if n_outputs > 2 else 0.0
    return n * binary_entropy(eps) + tail


def explainable_fraction_log2(ell: float, n: int, eps: float, n_outputs: int,
                              exact: bool = False) -> float:
    """log2 of the upper bound on the fraction of explainable tables (Theorem 2).

    With exact=True the Hamming-ball volume is computed exactly instead of
    through the entropy bound; both are valid upper bounds, the exact one tighter.
    """
    if exact:
        radius = int(math.floor(eps * n))
        return ell + hamming_ball_size(n, radius, n_outputs) - n * math.log2(n_outputs)
    return ell - n * budget_per_output(eps, n_outputs)
