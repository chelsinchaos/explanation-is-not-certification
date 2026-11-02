"""Pipeline scaffold for the pre-registered experiment (experiment/PREREGISTRATION.md).

The analysis half (AUROC with DeLong intervals, certified coverage under Theorem 5,
cost accounting) is implemented and tested on synthetic data.  The model half
(running Gemma-2-2B / Llama-3.2-1B, computing the three explanation residuals with
the open circuit-tracing library) is left as clearly marked stubs so that the
analysis cannot drift after data collection begins.

Run `python experiment/pipeline_scaffold.py --synthetic` to exercise the analysis
half end to end on synthetic scores.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from selcert import calibrate_bonferroni  # noqa: E402

SCORES = ["s1_error_node_mass", "s2_replacement_tv", "s3_reconstruction_error",
          "b1_max_softmax", "b2_entropy", "b3_self_consistency", "b4_verbalised_confidence",
          "b5_epistemic_risk"]
FORWARD_PASS_EQUIVALENTS = {  # to be measured, not assumed; placeholders from the paper's ranges
    "s1_error_node_mass": 50.0, "s2_replacement_tv": 60.0, "s3_reconstruction_error": 1.0,
    "b1_max_softmax": 1.0, "b2_entropy": 1.0, "b3_self_consistency": 10.0,
    "b4_verbalised_confidence": 2.0, "b5_epistemic_risk": 1.0,
}


# --------------------------------------------------------------------------- #
# Model half: STUBS. Fill these in before collecting data; do not touch the analysis half.
# --------------------------------------------------------------------------- #

def load_prompts(task: str, hint_condition: bool = False) -> List[Dict]:
    """Return a list of {"prompt": str, "answer": str} for task in {"A", "B", "C"}."""
    raise NotImplementedError("Build the prompt sets per PREREGISTRATION.md")


def run_model_and_scores(model_name: str, prompts: Sequence[Dict]) -> Dict[str, np.ndarray]:
    """Return {"correct": 0/1 array, score_name: array for each name in SCORES}.

    s1-s3 come from circuit-tracer attribution graphs; b1-b4 from the model's own
    outputs; b5 from the grouping-loss estimator (Melo et al. 2026).
    """
    raise NotImplementedError("Wire to circuit-tracer and the model runtime")


# --------------------------------------------------------------------------- #
# Analysis half: implemented.
# --------------------------------------------------------------------------- #

def delong_auroc_ci(scores: np.ndarray, errors: np.ndarray, level: float = 0.95):
    """AUROC for predicting error (higher score = more likely wrong) with a DeLong CI."""
    pos = scores[errors == 1]
    neg = scores[errors == 0]
    m, n = pos.size, neg.size
    if m == 0 or n == 0:
        return float("nan"), (float("nan"), float("nan"))
    # placement values (DeLong 1988), O(m n) is fine at these sizes
    cmp = (pos[:, None] > neg[None, :]).astype(float) + 0.5 * (pos[:, None] == neg[None, :])
    auc = cmp.mean()
    v10 = cmp.mean(axis=1)          # per positive
    v01 = cmp.mean(axis=0)          # per negative
    s10 = v10.var(ddof=1) if m > 1 else 0.0
    s01 = v01.var(ddof=1) if n > 1 else 0.0
    se = np.sqrt(s10 / m + s01 / n)
    from scipy.stats import norm
    z = norm.ppf(0.5 + level / 2)
    return float(auc), (float(auc - z * se), float(auc + z * se))


@dataclass
class CertRow:
    score: str
    alpha: float
    tau_hat: float | None
    test_coverage: float
    test_selective_error: float | None


def certify(score_name: str, grid_scores: np.ndarray, cal_scores: np.ndarray, cal_err: np.ndarray,
            test_scores: np.ndarray, test_err: np.ndarray, alpha: float, delta: float = 0.05,
            K: int = 50) -> CertRow:
    """Theorem 5 with a grid of K quantiles from the grid split (never the calibration sample)."""
    grid = np.unique(np.quantile(grid_scores, np.linspace(0, 1, K + 2)[1:-1]))
    res = calibrate_bonferroni(cal_scores, cal_err, grid, alpha, delta)
    if res.tau_hat is None:
        return CertRow(score_name, alpha, None, 0.0, None)
    acc = test_scores <= res.tau_hat
    cov = float(acc.mean())
    sel_err = float(test_err[acc].mean()) if acc.any() else None
    return CertRow(score_name, alpha, res.tau_hat, cov, sel_err)


def analyse(data: Dict[str, np.ndarray], seed: int, c_h: float = 5.0, pass_cost: float = 1e-3):
    rng = np.random.default_rng(seed)
    n = data["correct"].size
    perm = rng.permutation(n)
    g, c, t = perm[:400], perm[400:1200], perm[1200:2000]
    err = 1 - data["correct"]
    out = {"auroc": {}, "cert": [], "cost_per_request": {}}
    for name in SCORES:
        s = data[name]
        out["auroc"][name] = delong_auroc_ci(s[t], err[t])
        for alpha in (0.01, 0.05):
            out["cert"].append(certify(name, s[g], s[c], err[c], s[t], err[t], alpha))
        out["cost_per_request"][name] = FORWARD_PASS_EQUIVALENTS[name] * pass_cost
    # value identity (Proposition 2) at alpha = 0.05 against the best baseline
    cov = {r.score: r.test_coverage for r in out["cert"] if r.alpha == 0.05}
    best_base = max((k for k in cov if k.startswith("b")), key=lambda k: cov[k])
    out["value_vs_best_baseline"] = {
        k: (cov[k] - cov[best_base]) * c_h - (out["cost_per_request"][k] - out["cost_per_request"][best_base])
        for k in cov if k.startswith("s")
    }
    out["best_baseline"] = best_base
    return out


def synthetic_data(n: int = 2000, seed: int = 0) -> Dict[str, np.ndarray]:
    """Synthetic stand-in with a latent error signal, for exercising the analysis half only."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    correct = (z <= np.quantile(z, 0.9)).astype(int)
    noise = {k: 0.5 + 0.3 * i for i, k in enumerate(SCORES)}
    data = {"correct": correct}
    for k in SCORES:
        data[k] = z + noise[k] * rng.standard_normal(n)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    if not args.synthetic:
        raise SystemExit("Model half not implemented; run with --synthetic to test the analysis half.")
    res = analyse(synthetic_data(), seed=0)
    for k, (auc, ci) in res["auroc"].items():
        print(f"{k:28s} AUROC {auc:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
    for r in res["cert"]:
        print(f"{r.score:28s} alpha={r.alpha:.2f} tau={r.tau_hat} coverage={r.test_coverage:.3f} sel_err={r.test_selective_error}")
    print("best baseline:", res["best_baseline"])
    print("value per request vs best baseline (USD):", {k: round(v, 3) for k, v in res["value_vs_best_baseline"].items()})


if __name__ == "__main__":
    main()
