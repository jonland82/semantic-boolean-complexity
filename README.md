# Semantic Boolean Complexity

**Semantics finds where bounds are loose. Calibration decides what can be claimed.**

[Project site](https://jonland82.github.io/semantic-boolean-complexity/) ·
[Latest paper](papers/selection-breaks-calibration/selection-breaks-calibration.pdf) ·
[Core calibration paper](papers/learned-semantic-envelopes/learned-semantic-envelopes.pdf) ·
[Reproduce the analysis](experiments/four_bit/README.md)

Can the difficulty of representing a Boolean function be inferred from the
function itself, and can that information safely improve a mathematical bound?

This repository develops that question as a sequence of exact experiments and
mathematical results. Semantic truth-table features strongly predict minimum
Boolean formula size. Those features expose a structural picture based on
recursive decomposability, boundary obstruction, certificates, and algebraic
phase. They also locate exact complexity inside proved analytic envelopes.

The latest prospective experiment establishes an important limit. A learned
model successfully found loose constructive upper bounds, improving 82.5% of
120 untouched five-input cases by 0.8795 gate on average. But six selected
bounds became invalid, producing 95% coverage rather than the required 99%.
All six failures occupy the same one-gate-headroom boundary. The current
research problem is therefore not whether semantic learning contains useful
signal; it is how to calibrate that signal after model-based selection.

## The research arc

| Stage | Paper | Main result |
|---|---|---|
| 1 | [Semantic Structure Predicts](papers/semantic-structure-predicts/semantic-structure-predicts.pdf) | Twelve semantic invariants explain 77.7–84.3% of exact held-out complexity variation. |
| 2 | [A Semantic Construction Profile](papers/semantic-construction-profile/semantic-construction-profile.pdf) | A 66-coordinate profile raises held-out prediction to 94.7–97.3% and identifies recursive decomposability as the leading signal. |
| 3 | [Semantic Complexity Envelopes](papers/semantic-complexity-envelopes/semantic-complexity-envelopes.pdf) | Boundary lower bounds and explicit decision-tree and prime-cover constructions bracket exact complexity. |
| 4 | [Prediction-Calibrated Refinement](papers/learned-semantic-envelopes/learned-semantic-envelopes.pdf) | Learned intervals remove 70–82% of analytic-envelope width while retaining calibrated four-input coverage. |
| 5 | [Can Semantic Learning Tighten a Mathematical Bound?](papers/selective-semantic-bound-refinement/selective-semantic-bound-refinement.pdf) | A frozen cost-14 test finds real removable slack but rejects the proposed 99% calibration rule. |
| 6 | [When Selection Breaks Calibration](papers/selection-breaks-calibration/selection-breaks-calibration.pdf) | The six failures are an exact low-headroom boundary enriched by top-score selection. |

## Central formalism

For a Boolean function $f$, let

$$
Y(f)=K(f)+1
$$

be exact formula complexity in the leaf-aligned convention, and suppose an
explicit construction gives

$$
Y(f)\le U(f).
$$

The available construction headroom is

$$
h(f)=U(f)-Y(f)\ge 0.
$$

A semantic model produces a headroom score $z(f)$. Calibration supplies a
one-sided correction $q_{g(f)}$, indexed by an observable risk stratum. The
policy proposes

$$
r(f)=\bigl(z(f)-q_{g(f)}\bigr)_+,
\qquad
U_{\mathrm{sel}}(f)=U(f)-r(f).
$$

The refined endpoint is valid exactly when $r(f)\le h(f)$. Its violation is

$$
v(f)
=\bigl(Y(f)-U_{\mathrm{sel}}(f)\bigr)_+
=\bigl(r(f)-h(f)\bigr)_+.
$$

On an active $m$-headroom case,

$$
v(f)=\bigl(z(f)-(q_{g(f)}+m)\bigr)_+.
$$

This identity explains the prospective result: all six misses had $m=1$,
and the six largest scores in the matched ten-function boundary group were
exactly the six scores that crossed $q_{\mathrm{high}}+1$.

## Headline results

- Every one of the 65,536 four-input Boolean functions was synthesized exactly
  in NAND, NOR, and AND/OR/NOT formula languages.
- The expanded semantic construction profile explains 95.7%, 94.7%, and 97.3%
  of held-out exact-size variance in the three languages.
- The proved lower obstruction is the Khrapchenko boundary quantity

  $$
  B(f)=\frac{|E_{01}|^2}{|f^{-1}(0)|\,|f^{-1}(1)|}
  \le K_{\mathcal L}(f)+1.
  $$

- Four-input learned envelopes achieve 96.3–96.6% realized coverage at a
  nominal 95% level while removing 71.5–82.0% of analytic width on average.
- The frozen five-input cost-14 experiment improved 99 of 120 functions, but
  six of the 80 targeted cases violated the refined upper endpoint.
- One-headroom prevalence rose from 1/46 in the cost-13 targeted high-score
  calibration group to 9/78 at cost 14—a 5.31-fold enrichment.
- A retrospective one-gate reduction cap removes all six diagnostic failures
  while retaining 85.1% of the original tightening. This is a hypothesis for
  a future frozen test, not prospective evidence.

## What is established—and what remains open

**Proved generally**

- Boundary lower bounds and arbitrary-dimension constructive upper bounds.
- Intersection of an analytic envelope with a calibrated prediction interval
  preserves the calibration guarantee.
- Gap-normalized calibration gives a deterministic per-instance lower bound on
  fractional envelope tightening.
- The violation and selection-enrichment identities used in the failure audit.

**Exhaustively verified on four inputs**

- Exact minimum formula sizes for all functions in three gate languages.
- Sharp finite-domain envelope constants, prediction results, and calibration
  audits.

**Prospectively observed on five inputs**

- Semantic ranking locates useful construction slack.
- The frozen coarse-stratum correction does not achieve 99% coverage after
  top-score selection.

**Open**

- A selection-aware calibration rule that retains useful tightening.
- Reliable protection of zero- and one-headroom boundary cases.
- A new untouched exact layer or independent cohort for prospective validation.
- A theorem converting the learned structural pattern into a broader analytic
  complexity bound.

## Repository layout

```text
data/reference/                            exact reference data
experiments/four_bit/                      complete four-input synthesis and analysis
experiments/five_bit_pilot/                cross-dimension feasibility tests
experiments/five_bit_next_step/            five-input construction study
experiments/residual_slack_refinement/     cost-13 prospective refinement
experiments/selective_slack_refinement/    frozen cost-14 test and failure audit
papers/                                    six-paper research sequence
tests/                                     reproducibility and consistency checks
```

## Reproduction

Python 3.10 or newer is recommended.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m pytest -q
```

Run commands from the repository root. The complete four-input reproduction
sequence is in [`experiments/four_bit/README.md`](experiments/four_bit/README.md).
The cost-14 artifacts, structural audit, and local selection-aware diagnostics
are documented in
[`experiments/selective_slack_refinement/README.md`](experiments/selective_slack_refinement/README.md).
Downstream analysis runs locally. Regenerating the full exact five-input
cost-14 layer is a substantial multi-hour computation and was performed on AWS.

## Papers

- [When Selection Breaks Calibration: Constructive Collapse in Learned
  Boolean-Complexity Bounds](papers/selection-breaks-calibration/selection-breaks-calibration.pdf)
- [Can Semantic Learning Tighten a Mathematical
  Bound?](papers/selective-semantic-bound-refinement/selective-semantic-bound-refinement.pdf)
- [Prediction-Calibrated Refinement of Analytic Bounds with an Application to
  Boolean Formula Complexity](papers/learned-semantic-envelopes/learned-semantic-envelopes.pdf)
- [Semantic Complexity Envelopes for Exact Boolean Formula
  Size](papers/semantic-complexity-envelopes/semantic-complexity-envelopes.pdf)
- [A Semantic Construction Profile for Exact Boolean Formula
  Complexity](papers/semantic-construction-profile/semantic-construction-profile.pdf)
- [Semantic Structure Predicts Exact Boolean Formula
  Complexity](papers/semantic-structure-predicts/semantic-structure-predicts.pdf)

Code is MIT licensed. Original papers, reports, and figures are CC BY 4.0;
see [`LICENSE-CONTENT.md`](LICENSE-CONTENT.md).
