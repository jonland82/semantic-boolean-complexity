# Semantic Boolean Complexity

**Theory sets the range. Semantics finds the answer.**

[Project site](https://jonland82.github.io/semantic-boolean-complexity/) ·
[Main paper](papers/learned-semantic-envelopes/learned-semantic-envelopes.pdf) ·
[Reproduce the analysis](experiments/four_bit/README.md)

Can exact Boolean formula complexity be predicted from the function itself,
rather than from a formula that computes it—and can that prediction tighten a
proved analytic bound without weakening its guarantee?

This repository studies that question exhaustively for all 65,536 four-input
Boolean functions. Exact minimum formula sizes are computed in NAND, NOR, and
AND/OR/NOT languages and compared with input-renaming-invariant semantic
descriptors. The current experiments identify a compact structural picture:

1. Khrapchenko-style boundary complexity supplies a lower obstruction.
2. Decision-tree decomposability supplies an upper construction.
3. Certificates, algebraic support, and Fourier phase help locate exact cost
   between those bounds.

## Mathematical framework

Let $Y(x)$ be an unknown quantity for an instance $x$. Suppose analysis gives
a valid outer envelope

$$
I_0(x)=[L(x),H(x)], \qquad L(x)\leq Y(x)\leq H(x),
$$

and let $\widehat Y(x)$ be any learned point prediction. Calibration supplies
one-sided error allowances $E^-(x),E^+(x)\geq 0$, defining

$$
C(x)=[\widehat Y(x)-E^-(x),\widehat Y(x)+E^+(x)].
$$

The prediction-calibrated envelope is their intersection:

$$
I_*(x)=I_0(x)\cap C(x)
=\left[
\max\left(L(x),\widehat Y(x)-E^-(x)\right),
\min\left(H(x),\widehat Y(x)+E^+(x)\right)
\right].
$$

This operator separates validity from tightness. Because $I_*(x)\subseteq
I_0(x)$, it can never widen the analytic bound. Because $Y(x)\in I_0(x)$,
the refined envelope contains $Y(x)$ whenever the calibrated interval $C(x)$
does. It therefore inherits the statistical, finite-domain, or universal
coverage guarantee used to construct $E^-$ and $E^+$.

The stronger result calibrates error relative to the analytic gap
$W(x)=H(x)-L(x)$. Clip $\widehat Y(x)$ to $[L(x),H(x)]$, define its normalized
position

$$
p(x)=\frac{\widehat Y(x)-L(x)}{W(x)},
$$

and calibrate the one-sided normalized errors

$$
S^-(x)=\frac{(\widehat Y(x)-Y(x))_+}{W(x)},
\qquad
S^+(x)=\frac{(Y(x)-\widehat Y(x))_+}{W(x)},
$$

where $(z)_+=\max(z,0)$. Let $q^-$ and $q^+$ be bounds or calibrated
quantiles for these scores. The refined envelope becomes

$$
I_q(x)=\left[
\max\left(L(x),\widehat Y(x)-q^-W(x)\right),
\min\left(H(x),\widehat Y(x)+q^+W(x)\right)
\right].
$$

For every nondegenerate analytic envelope,

$$
\frac{\mathrm{width}(I_q(x))}{W(x)}
=\min\left(p(x),q^-\right)+\min\left(1-p(x),q^+\right)
\leq \min\left(1,q^-+q^+\right).
$$

Consequently, every instance loses at least
$\max(0,1-q^--q^+)$ of its original analytic width. If $q^-$ and $q^+$ are
split-conformal quantiles with tail errors $\alpha^-$ and $\alpha^+$, then
$I_q$ covers $Y$ with probability at least $1-\alpha^--\alpha^+$. Exhaustive
or universal score bounds give the corresponding deterministic guarantee.

### Application to Boolean formula complexity

For a Boolean function
$f:\lbrace 0,1\rbrace^n\to\lbrace 0,1\rbrace$ and gate library $\mathcal L$,
the target is the minimum tree-formula gate count

$$
K_{\mathcal L}(f)=
\min_{e:\,\mathrm{eval}_{\mathcal L}(e)=f}|e|.
$$

The framework uses $Y_{\mathcal L}(f)=K_{\mathcal L}(f)+1$ to align the target
with the leaf-count lower bound. If $Z=f^{-1}(0)$, $O=f^{-1}(1)$, and $E_{01}$
is the set of Hamming-neighbor pairs across which $f$ changes value, the
Khrapchenko quantity

$$
B(f)=\frac{|E_{01}|^2}{|Z||O|}
$$

gives the common lower bound

$$
B(f)\leq K_{\mathcal L}(f)+1.
$$

Constructive upper bounds come from $U(f)$, the minimum number of leaves in a
deterministic decision tree for $f$. On the exhaustively verified four-input
domain, the resulting analytic sandwiches are

$$
B(f)\leq K_{\mathrm{NAND/NOR}}(f)+1
\leq 6+\frac{9}{4}(U(f)-1),
$$

and

$$
B(f)\leq K_{\mathrm{AON}}(f)+1
\leq \min\left(3+\frac{13}{9}(U(f)-1),Q_{\min}(f)\right),
$$

where $Q_{\min}(f)$ is the upper-bound value from the best explicit
prime-cover construction found for AND/OR/NOT. These four-input coefficients
are finite-domain facts; the paper also gives looser arbitrary-dimension
constructions.

The learned model predicts where the exact value lies inside this proved
sandwich. For $L_{\mathcal L}(f)<H_{\mathcal L}(f)$, define

$$
\rho_{\mathcal L}(f)=
\frac{K_{\mathcal L}(f)+1-L_{\mathcal L}(f)}
{H_{\mathcal L}(f)-L_{\mathcal L}(f)},
$$

then predict

$$
\widehat Y_{\mathcal L}(f)
=L_{\mathcal L}(f)
+\widehat\rho(\Phi(f))
\bigl(H_{\mathcal L}(f)-L_{\mathcal L}(f)\bigr).
$$

Here $\Phi(f)$ contains semantic and constructive invariants of the function,
not a candidate formula. Calibration converts this prediction into a qualified
interval, and intersection with the analytic sandwich preserves the original
theory while removing much of its unresolved width.

## Headline results

- The original 12-coordinate descriptor explains 77.7% of exact NAND/NOR
  variance and 84.3% of AND/OR/NOT variance under strict descriptor holdout.
- The expanded 66-coordinate model reaches held-out $R^2$ values of 95.7%, 94.7%,
  and 97.3%.
- A compact 16-coordinate lower--upper sandwich model retains 94.6%, 93.0%,
  and 95.4% gate-count $R^2$ and explains 79.6%, 73.8%, and 61.0% of normalized
  position within the sandwich. The AND/OR/NOT envelope includes an exact
  minimum-cost prime-cover construction.
- A learned-envelope analysis centers calibrated and exhaustively audited
  intervals on the compact model's predicted gate count, then intersects them
  with the classical semantic envelope. At nominal 95% class coverage, the
  intervals cover 96.3%, 96.6%, and 96.4% of functions while removing 82.0%,
  79.2%, and 71.5% of classical width on average. Maximum-residual intervals
  cover the full four-input universe while removing 54.7%, 57.1%, and 47.9%.
- Gap-normalized calibration turns that empirical tightening into a general
  theorem: if the calibrated one-sided error fractions are $q^-$ and
  $q^+$, every noncollapsed analytic envelope shrinks by at least
  $1-q^--q^+$, while retaining the corresponding statistical,
  finite-domain, or universal coverage guarantee. At nominal 95% coverage,
  this certifies at least 81.4%, 79.0%, and 70.8% pointwise shrinkage for the
  held-out NAND, NOR, and AND/OR/NOT envelopes.
- In a frozen local five-input cost-11 test, semantic conditional error margins
  cover 100%, 100%, and 92.5% of sampled NAND, NOR, and AND/OR/NOT functions
  while removing 94.7%, 93.8%, and 71.9% of the applicable general envelope.

## Repository layout

```text
data/reference/                         three-input exact reference data
experiments/four_bit/scripts/           synthesis and analysis programs
experiments/four_bit/artifacts/         exact targets and derived tables
experiments/four_bit/figures/           publication figures
experiments/four_bit/reports/           generated experiment reports
experiments/five_bit_pilot/             local cross-dimension feasibility test
papers/semantic-structure-predicts/     original four-page paper
papers/semantic-construction-profile/   four-page follow-up paper
papers/semantic-complexity-envelopes/   four-page bounds paper
papers/learned-semantic-envelopes/      calibrated bound-refinement short paper
```

## Setup

Python 3.10 or newer is recommended.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e .
```

Run commands from the repository root. See
[`experiments/four_bit/README.md`](experiments/four_bit/README.md) for the full
reproduction sequence. The downstream analyses use only local computation;
no model API or cloud service is required.

## Papers

- **Main paper:** [Prediction-Calibrated Refinement of Analytic Bounds with an
  Application to Boolean Formula Complexity](papers/learned-semantic-envelopes/learned-semantic-envelopes.pdf)
- [Semantic Complexity Envelopes for Exact Boolean Formula
  Size](papers/semantic-complexity-envelopes/semantic-complexity-envelopes.pdf)
- [A Semantic Construction Profile for Exact Boolean Formula
  Complexity](papers/semantic-construction-profile/semantic-construction-profile.pdf)
- [Semantic Structure Predicts Exact Boolean Formula
  Complexity](papers/semantic-structure-predicts/semantic-structure-predicts.pdf)

Code is MIT licensed. Original papers, reports, and figures are CC BY 4.0;
see [`LICENSE-CONTENT.md`](LICENSE-CONTENT.md).
