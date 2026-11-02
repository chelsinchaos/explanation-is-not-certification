"""Generate every figure and every simulation number in the paper.

Usage:  python scripts/make_figures.py [--fast]

Writes paper/figures/*.pdf and paper/figures/results.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker
import matplotlib.pyplot as plt

from selcert.bounds import budget_per_output, hamming_ball_size, entropy_bound_log2
from selcert.simulate import study_A, study_B, study_C, rows_to_dicts

OUT = os.path.join(os.path.dirname(__file__), "..", "paper", "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 150, "savefig.bbox": "tight", "axes.spines.top": False,
    "axes.spines.right": False, "font.family": "DejaVu Sans",
})
C = ["#1b4f72", "#c0392b", "#117a65", "#7d6608", "#6c3483", "#7f8c8d"]


def fig_budget():
    """Figure 1: worst-case explanation budget per audited output, B(eps,|Y|)."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), constrained_layout=True)
    ax = axes[0]
    eps = np.linspace(0.0, 0.5, 501)
    for i, q in enumerate([2, 4, 16, 2 ** 17]):
        lab = {2: "|Y| = 2", 4: "|Y| = 4", 16: "|Y| = 16", 2 ** 17: "|Y| = 2^17 (token vocabulary)"}[q]
        ax.plot(eps, [budget_per_output(e, q) for e in eps], color=C[i], label=lab)
    ax.set_xlabel("residual tolerance ε")
    ax.set_ylabel("B(ε, |Y|)  [bits per audited output]")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 30)
    ax.set_title("a  Worst-case budget per output", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    ax = axes[1]
    n = 10_000
    eps_grid = np.array([0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3])
    exact = np.array([hamming_ball_size(n, int(math.floor(e * n)), 2) for e in eps_grid])
    bound = np.array([entropy_bound_log2(n, e, 2) for e in eps_grid])
    ax.plot(eps_grid, bound - exact, "o-", color=C[0])
    ax.set_xlabel("residual tolerance ε")
    ax.set_ylabel("bound minus exact count [bits]")
    ax.set_ylim(0, 8)
    ax.set_title("b  Slack of the bound (n = 10,000, |Y| = 2)", loc="left", fontweight="bold")
    fig.savefig(os.path.join(OUT, "fig_budget.pdf")); fig.savefig(os.path.join(OUT, "fig_budget.png"), dpi=200)
    plt.close(fig)
    return {"n": n, "eps": eps_grid.tolist(), "exact_bits": (n - exact).tolist(),
            "entropy_bound_bits": (n - bound).tolist(), "slack_bits": (bound - exact).tolist()}


def _mticks(ax, ms):
    ax.set_xscale("log")
    ax.set_xticks(ms)
    ax.set_xticklabels([str(m) for m in ms])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def fig_simulation(resA, resC):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), constrained_layout=True)
    ms = sorted({r["m"] for r in resA})
    # a: coverage vs m
    ax = axes[0]
    for j, alpha in enumerate([0.01, 0.05]):
        rows = sorted([r for r in resA if r["variant"] == "bonferroni_fixed_grid" and r["alpha"] == alpha], key=lambda r: r["m"])
        ax.plot([r["m"] for r in rows], [r["mean_coverage"] for r in rows], "o-", color=C[j], ms=4,
                label=f"certified, α = {alpha:g}")
        ax.axhline(rows[0]["oracle_coverage"], color=C[j], lw=0.8, ls=":")
        ax.text(8500, rows[0]["oracle_coverage"] + 0.012, f"oracle, α = {alpha:g}", fontsize=6.5, color=C[j], ha="right")
    _mticks(ax, ms)
    ax.set_ylim(0, 0.55)
    ax.set_xlabel("calibration size m")
    ax.set_ylabel("mean true coverage C(τ̂)")
    ax.set_title("a  Coverage vs calibration size", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=7, loc="center left")
    # b: coverage vs AUROC
    ax = axes[1]
    rows = sorted(resC, key=lambda r: r["auroc"])
    ax.plot([r["auroc"] for r in rows], [r["mean_certified_coverage"] for r in rows], "o-", color=C[0], ms=4,
            label="certified, m = 2000, α = 0.05")
    ax.plot([r["auroc"] for r in rows], [r["oracle_coverage"] for r in rows], "s:", color=C[5], ms=4,
            label="oracle")
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("score quality (AUROC for predicting error)")
    ax.set_ylabel("mean true coverage C(τ̂)")
    ax.set_title("b  Coverage vs score quality", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.savefig(os.path.join(OUT, "fig_simulation.pdf")); fig.savefig(os.path.join(OUT, "fig_simulation.png"), dpi=200)
    plt.close(fig)


def table_violation(resA):
    """Table 1 of the paper: violation rates by procedure, alpha and m (LaTeX body).
    Rows are calibration sizes; columns are (procedure, alpha) pairs."""
    ms = sorted({r["m"] for r in resA})
    variants = [("bonferroni_fixed_grid", "Theorem~\\ref{thm:selective}"),
                ("bonferroni_sample_quantile_grid", "sample-quantile grid"),
                ("uncorrected_fixed_grid", "no multiplicity correction")]
    idx = {(r["variant"], r["alpha"], r["m"]): r for r in resA}
    def cell(variant, alpha, m):
        r = idx[(variant, alpha, m)]
        v = r["violation_rate"]
        vs = "$<\\!10^{-4}$" if v == 0 else f"{v:.4f}"
        return f"{vs} ({100 * r['defer_all_rate']:.0f}\\%)"
    lines = ["\\begin{tabular}{r" + "c" * 6 + "}", "\\toprule",
             " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{name}}}" for _, name in variants) + " \\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
             "$m$ & " + " & ".join("$\\alpha=0.01$ & $\\alpha=0.05$" for _ in variants) + " \\\\",
             "\\midrule"]
    for m in ms:
        lines.append(f"{m} & " + " & ".join(cell(v, a, m) for v, _ in variants for a in (0.01, 0.05)) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(OUT, "table_violation.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="fewer replications (smoke test)")
    args = ap.parse_args()
    t0 = time.time()
    repsA = 2000 if args.fast else 10000
    repsB = 200_000 if args.fast else 2_000_000
    repsC = 300 if args.fast else 2000
    truthC = 400_000 if args.fast else 4_000_000
    results = {"budget": fig_budget()}
    print("study A ...", flush=True)
    resA = rows_to_dicts(study_A(reps=repsA))
    print("study B ...", flush=True)
    resB = rows_to_dicts(study_B(reps=repsB))
    print("study C ...", flush=True)
    resC = rows_to_dicts(study_C(reps=repsC, truth_samples=truthC))
    results.update({"study_A": resA, "study_B": resB, "study_C": resC,
                    "fast": args.fast, "seconds": time.time() - t0})
    fig_simulation(resA, resC)
    table_violation(resA)
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    print(f"done in {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
