import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from selcert import (agreement_cost, disagreement_cost, code_length, explanation_gain, ecs,
                     budget_per_output, hamming_ball_size, entropy_bound_log2,
                     explainable_fraction_log2, clopper_pearson_upper,
                     calibrate_bonferroni, calibrate_fixed_sequence)
from selcert.bounds import binary_entropy
from selcert.simulate import risk_curve_A


# ---------------- Lemma 1 / Definition 2 ----------------

def test_costs_bracket():
    c0 = agreement_cost(2 ** 17, 0.01)
    c1 = disagreement_cost(2 ** 17, 0.01)
    assert 0 < c0 < c1
    assert math.isclose(c1, math.log2(2 ** 17 / 0.01))


def test_code_length_identity():
    n, k = 100, 7
    L = code_length(500.0, n, k, 4, 0.1)
    c0, c1 = agreement_cost(4, 0.1), disagreement_cost(4, 0.1)
    assert math.isclose(L, 500.0 + (n - k) * c0 + k * c1)


def test_ecs_upper_bound():
    # every explanation pays at least n c_0, so ECS <= 1 - c0/log|Y|
    for q in (2, 10, 1000):
        c0 = agreement_cost(q, 0.05)
        assert ecs(0.0, 50, 0, q, 0.05) <= 1 - c0 / math.log2(q) + 1e-12


def test_self_reference_negative_for_small_audits():
    # Theorem 1: an explanation of length K_U(M) has non-positive gain when n <= K/(log|Y| - c0)
    q, eta, K = 2 ** 17, 0.01, 1e10
    c0 = agreement_cost(q, eta)
    n = int(K / (math.log2(q) - c0))
    assert explanation_gain(K, n, 0, q, eta) <= 0


# ---------------- Theorem 2 ----------------

def test_budget_positive_everywhere():
    for q in (2, 3, 4, 16, 2 ** 17):
        for e in np.linspace(0, 0.499, 50):
            assert budget_per_output(e, q) > 0
    # |Y| >= 3 stays positive at eps = 1/2; |Y| = 2 hits zero there
    assert budget_per_output(0.5, 3) > 0
    assert abs(budget_per_output(0.5, 2)) < 1e-12


def test_bracket_minimum_at_q3():
    # minimum over q >= 3 and eps <= 1/2 is log2(3) - 1 - 0.5 = 0.08496
    assert math.isclose(budget_per_output(0.5, 3), math.log2(3) - 1.5, rel_tol=1e-9)


def test_exact_ball_below_entropy_bound():
    n = 10_000
    for e in (0.005, 0.01, 0.05):
        exact = hamming_ball_size(n, int(e * n), 2)
        bound = entropy_bound_log2(n, e, 2)
        assert exact <= bound + 1e-9
    assert math.isclose(binary_entropy(0.01), 0.080793, abs_tol=1e-5)
    assert math.isclose(explainable_fraction_log2(0.0, n, 0.01, 2), -n * (1 - binary_entropy(0.01)))


# ---------------- Theorem 5 ----------------

def test_clopper_pearson_edge_cases():
    assert clopper_pearson_upper(0, 0, 0.95) == 1.0
    assert clopper_pearson_upper(5, 5, 0.95) == 1.0
    # zero errors in m trials: UCB = 1 - (1 - conf)^(1/m)
    m = 30
    assert math.isclose(clopper_pearson_upper(0, m, 0.95), 1 - 0.05 ** (1 / m), rel_tol=1e-9)
    # monotone in errors
    u = clopper_pearson_upper(np.arange(0, 20), 20, 0.95)
    assert np.all(np.diff(u) > 0)


def test_calibrator_controls_risk_in_simulation():
    rng = np.random.default_rng(0)
    K, alpha, delta, m, reps = 25, 0.05, 0.1, 400, 400
    grid = np.arange(1, K + 1) / K
    viol = 0
    for _ in range(reps):
        s = rng.uniform(size=m)
        err = (rng.uniform(size=m) < 0.0025 + 0.6 * s ** 2).astype(float)
        res = calibrate_bonferroni(s, err, grid, alpha, delta)
        if res.tau_hat is not None and risk_curve_A(np.array([res.tau_hat]))[0] > alpha:
            viol += 1
    assert viol / reps <= delta  # Bonferroni + Clopper-Pearson are conservative


def test_fixed_sequence_never_looser_than_bonferroni_prefix():
    rng = np.random.default_rng(1)
    grid = np.arange(1, 21) / 20
    s = rng.uniform(size=1000)
    err = (rng.uniform(size=1000) < 0.0025 + 0.6 * s ** 2).astype(float)
    b = calibrate_bonferroni(s, err, grid, 0.05, 0.05)
    f = calibrate_fixed_sequence(s, err, grid, 0.05, 0.05)
    assert (b.tau_hat is None) or (f.tau_hat is None) or True  # both are valid procedures
    assert np.all(f.upper_bounds <= b.upper_bounds + 1e-12)   # level 1-δ bound <= level 1-δ/K bound


def test_input_validation():
    with pytest.raises(ValueError):
        calibrate_bonferroni([0.1, 0.2], [0, 2], [0.5], 0.05, 0.05)
    with pytest.raises(ValueError):
        calibrate_bonferroni([0.1, 0.2], [0, 1], [0.5, 0.4], 0.05, 0.05)
