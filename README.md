# Explanation is not certification

**Information-theoretic limits of interpretability-based assurance for machine learning systems**

Chelsea McElveen · LogNovations Holdings LLC · preprint, November 2026

[**Paper (PDF)**](paper/main.pdf) · [LaTeX source](paper/main.tex) · [Pre-registered experiment](experiment/PREREGISTRATION.md)

---

Regulated deployments of machine learning keep a human reviewer in the loop because no one can certify a model's outputs from the outside. A widely held hope is that interpretability will remove that reviewer: if the model's internal computation can be seen, its outputs can be certified. This paper gives that hope a formal test.

An explanation is defined as a two-part code for a model's behaviour on an audit set and scored by the bits it saves. Five limits follow:

1. **Self-reference.** Offering the model itself as its explanation has non-positive gain on any audit smaller than the model; transmitting its internal state has negative gain always.
2. **No universal explainer.** No fixed explanation budget covers all behaviours, so every fixed-budget standard contains an unavoidable defer outcome.
3. **Uncomputability.** The optimal explanation score is uncomputable; a checkable standard must fix a decidable explanation language.
4. **Independence.** Faithfulness and correctness are independent; an explanation can never replace a specification or a label.
5. **Chain of thought.** Read as an explanation, a chain of thought has negative gain and, where tested, omits an inserted cause of the output on most inputs.

What *can* be certified: with any score fixed in advance and an i.i.d. calibration sample, a finite-sample bound on the error rate among auto-accepted outputs (a classical selective-classification guarantee). The guarantee does not depend on interpretability; within it, an explanation residual is worth exactly the human reviews it removes. Under the same accounting, an attribution graph with the model's attention patterns frozen in is a second model, not an explanation; its legitimate role is as a score.

The paper states the implications for the EU AI Act (Articles 13 and 14, harmonised standards), SR 11-7 model-risk management, and the EASA / ED-324 aviation framework, and pre-registers the one experiment that measures whether an explanation residual adds certified coverage.

## Repository layout

```
paper/            LaTeX source, bibliography, compiled PDF, generated figures and table
src/selcert/      Python package: code lengths (Lemma 1), counting bounds (Theorem 2),
                  certified selective deployment (Theorem 5), simulation studies
scripts/          make_figures.py regenerates every figure, table and number in the paper
tests/            pytest suite (11 tests)
experiment/       pre-registration of the attribution-graph experiment and a pipeline scaffold
```

## Reproduce

```bash
python3 -m pip install -r requirements.txt
make test        # 11 tests, ~2 s
make figures     # all simulations + figures + Table 1, ~30 s on one CPU core
make paper       # pdflatex + bibtex via latexmk
```

Every number in the paper is produced by `scripts/make_figures.py` and written to `paper/figures/results.json`; the simulations use fixed seeds.

## Using the calibrator

```python
import numpy as np
from selcert import calibrate_bonferroni

# scores (lower = more trustworthy), 0/1 losses from your reviewers, on an i.i.d. calibration sample
res = calibrate_bonferroni(scores, losses, grid=np.linspace(0, 1, 51)[1:], alpha=0.05, delta=0.05)
print(res.tau_hat, res.empirical_coverage)   # tau_hat is None if nothing can be certified
```

The hypotheses matter: the grid, the score and the model must be fixed before the calibration sample is drawn, and the sample must be i.i.d. from the deployment distribution. Section 5 of the paper shows what goes wrong when the grid is taken from the sample's own quantiles.

## Citation

```bibtex
@misc{mcelveen2026explanation,
  author = {McElveen, Chelsea},
  title  = {Explanation is not certification: information-theoretic limits of
            interpretability-based assurance for machine learning systems},
  year   = {2026},
  note   = {Preprint},
  url    = {https://github.com/chelsinchaos/explanation-is-not-certification}
}
```

See `CITATION.cff` for machine-readable metadata.

## Licence

Code: MIT (see `LICENSE`). Paper text and figures: CC BY 4.0 (see `paper/LICENSE`).
