"""Two-part code length of an explanation (Definition 1, Lemma 1).

Notation follows the paper.  Logarithms are base 2.

    c_0 = log2 1 / (1 - eta + eta/|Y|)      agreement cost
    c_1 = log2 |Y| / eta                     disagreement cost

For a deterministic model and a deterministic explanation that disagrees with
the model on k of n audited inputs, the correction term equals
(n - k) c_0 + k c_1  (Lemma 1).
"""
from __future__ import annotations

import math


def agreement_cost(n_outputs: int, eta: float) -> float:
    """c_0: bits paid per audited input when the explanation is right."""
    _check(n_outputs, eta)
    return -math.log2(1.0 - eta + eta / n_outputs)


def disagreement_cost(n_outputs: int, eta: float) -> float:
    """c_1: bits paid per audited input when a deterministic explanation is wrong."""
    _check(n_outputs, eta)
    return math.log2(n_outputs / eta)


def code_length(length_bits: float, n_audit: int, n_disagree: int,
                n_outputs: int, eta: float) -> float:
    """L_D(E) for a deterministic model and deterministic explanation (Lemma 1)."""
    if not 0 <= n_disagree <= n_audit:
        raise ValueError("n_disagree must lie in [0, n_audit]")
    c0 = agreement_cost(n_outputs, eta)
    c1 = disagreement_cost(n_outputs, eta)
    return length_bits + (n_audit - n_disagree) * c0 + n_disagree * c1


def explanation_gain(length_bits: float, n_audit: int, n_disagree: int,
                     n_outputs: int, eta: float) -> float:
    """G_D(E) = n log2|Y| - L_D(E)."""
    null = n_audit * math.log2(n_outputs)
    return null - code_length(length_bits, n_audit, n_disagree, n_outputs, eta)


def ecs(length_bits: float, n_audit: int, n_disagree: int,
        n_outputs: int, eta: float) -> float:
    """Explanation compression score ECS_D(E) = G_D(E) / (n log2|Y|)  (<= 1)."""
    null = n_audit * math.log2(n_outputs)
    return explanation_gain(length_bits, n_audit, n_disagree, n_outputs, eta) / null


def _check(n_outputs: int, eta: float) -> None:
    if n_outputs < 2:
        raise ValueError("|Y| must be at least 2")
    if not 0.0 < eta < 1.0:
        raise ValueError("eta must lie in (0, 1)")
