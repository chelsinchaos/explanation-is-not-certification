# Pre-registration: do explanation residuals raise certified coverage?

Fixed on 2 November 2026, with version 1 of the preprint. This file mirrors
Appendix B of the paper; where they differ, the paper governs.

## Question

Does an explanation residual predict a model's errors well enough to raise
certified coverage (Theorem 5) over cheap uncertainty scores, at a compute cost
below the review cost it saves?

## Models

- Gemma-2-2B with Gemma Scope transcoders
- Llama-3.2-1B

Both are supported by the open circuit-tracing library (Anthropic, 2025) and the
Neuronpedia front end.

## Tasks (deterministic ground truth)

| id | task | prompts |
|----|------|---------|
| A | factual recall, country to capital | 2,000 |
| B | two-digit addition | 2,000 |
| C | four-way MMLU subset | 2,000 |
| D | tasks A–C with injected spurious hints (Chen et al. 2025 protocol) | 3 × 2,000 |

Condition D changes the mechanism without changing the task.

## Scores

Explanation residuals:

- `s1` error-node mass: share of attribution to the target logit that flows from error nodes
- `s2` replacement-model total-variation distance (graph features only, error nodes zeroed)
- `s3` final-layer transcoder reconstruction error norm (one forward pass)

Baselines:

- `b1` maximum softmax probability
- `b2` predictive entropy
- `b3` self-consistency disagreement over 10 samples at temperature 0.7
- `b4` verbalised confidence
- `b5` the epistemic-risk (grouping-loss) estimator of Melo, Varoquaux and le Morvan (ICLR 2026)

## Protocol

- Each task's 2,000 prompts are split 400 (grid) / 800 (calibration) / 800 (test), over five seeds.
- The grid for Theorem 5 is K = 50 score quantiles computed on the grid split only, never on the calibration sample.
- Reported per score and task: AUROC for predicting error with DeLong 95% intervals; certified coverage at α ∈ {1%, 5%}, δ = 5%; compute cost in forward-pass equivalents.
- At α = 1% and m = 800 the procedure is expected to defer everything unless a task's error rate is far below 1%; the condition is retained to document this.

## Hypotheses

- **H1** AUROC(s_i) exceeds the best baseline's for some i on some task, interval excluding zero.
- **H2** Certified coverage gain ΔC(5%) > 0 for some s_i on some task, interval excluding zero.
- **H3** ΔC · c_h exceeds Δc_s at c_h = 5 USD.
- **H4** On condition D, some s_i beats every baseline (AUROC).

## Predicted outcomes (written before any data)

| hypothesis | probability |
|------------|-------------|
| H1 on at least one of A–C | 30–45% |
| H1 on all three | 10–20% |
| H2 on at least one task | 25–40% |
| H3 given H2 | 30–50% |
| H4 | 50–65% |

Reasons: sparse-dictionary error norms are largely predictable from the
activations (Engels, Riggs, Tegmark 2024), so an in-distribution residual may add
little beyond confidence; coverage gains are smaller than AUROC gains after the
multiplicity correction; attribution costs 10–100 forward passes and only `s3`
is cheap; a hint is a mechanism change, which a residual defined against
trusted-data mechanisms is built to see and a confidence score is not.

## Reporting

Both outcomes will be reported in full. A negative result establishes that
explanation residuals are not useful deferral signals in distribution for these
models and tasks, and leaves the certification offer of Theorem 5 standing on
existing scores.

## Analysis code

`experiment/pipeline_scaffold.py` wires the scoring, AUROC and certification
steps to `selcert`; the model-specific steps (running the models, computing the
three residuals with the circuit-tracing library) are stubs to be filled before
any data are collected. The scaffold itself will not be changed after data
collection begins except to fix bugs, which will be logged in the commit history.
