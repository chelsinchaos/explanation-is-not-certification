"""selcert: explanation compression scores and certified selective deployment.

Companion code for

    McElveen, C. (2026). Explanation is not certification: information-theoretic
    limits of interpretability-based assurance for machine learning systems.

Modules
-------
code_length  Two-part code length of an explanation (Lemma 1), explanation gain
             and the explanation compression score (Definition 2).
bounds       Counting bounds on explainable behaviour tables (Theorem 2) and the
             per-output budget function B(eps, |Y|).
calibrate    Finite-sample certified selective deployment (Theorem 5):
             Clopper-Pearson + Bonferroni, and Learn-then-Test fixed-sequence.
simulate     Simulation studies reported in the paper.
"""

from .code_length import (agreement_cost, disagreement_cost, code_length,
                          explanation_gain, ecs)
from .bounds import (hamming_ball_size, entropy_bound_log2, budget_per_output,
                     explainable_fraction_log2)
from .calibrate import (clopper_pearson_upper, calibrate_bonferroni,
                        calibrate_fixed_sequence, CalibrationResult)

__all__ = [
    "agreement_cost", "disagreement_cost", "code_length", "explanation_gain",
    "ecs", "hamming_ball_size", "entropy_bound_log2", "budget_per_output",
    "explainable_fraction_log2", "clopper_pearson_upper",
    "calibrate_bonferroni", "calibrate_fixed_sequence", "CalibrationResult",
]
__version__ = "0.1.0"
