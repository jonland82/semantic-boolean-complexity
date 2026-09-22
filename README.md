# Semantic Boolean Complexity

Can exact Boolean formula complexity be predicted from the function itself,
rather than from a formula that computes it?

This repository studies that question exhaustively for all 65,536 four-input
Boolean functions. Exact minimum formula sizes are computed in NAND, NOR, and
AND/OR/NOT languages and compared with input-renaming-invariant semantic
descriptors. The current experiments identify a compact structural picture:

1. Khrapchenko-style boundary complexity supplies a lower obstruction.
2. Decision-tree decomposability supplies an upper construction.
3. Certificates, algebraic support, and Fourier phase help locate exact cost
   between those bounds.

## Headline results

- The original 12-coordinate descriptor explains 77.7% of exact NAND/NOR
  variance and 84.3% of AND/OR/NOT variance under strict descriptor holdout.
- The expanded 66-coordinate model reaches held-out R2 values of 95.7%, 94.7%,
  and 97.3%.
- A compact 16-coordinate lower--upper sandwich model retains 94.6%, 93.0%,
  and 95.4% gate-count R2 and explains 79.6%, 73.8%, and 61.0% of normalized
  position within the sandwich. The AND/OR/NOT envelope includes an exact
  minimum-cost prime-cover construction.
- A learned-envelope analysis centers calibrated and exhaustively audited
  intervals on the compact model's predicted gate count, then intersects them
  with the classical semantic envelope. At nominal 95% class coverage, the
  intervals cover 96.3%, 96.6%, and 96.4% of functions while removing 82.0%,
  79.2%, and 71.5% of classical width on average. Maximum-residual intervals
  cover the full four-input universe while removing 54.7%, 57.1%, and 47.9%.
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
experiments/five_bit_pilot/              local cross-dimension feasibility test
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

- [Semantic Structure Predicts Exact Boolean Formula Complexity](papers/semantic-structure-predicts/note.pdf)
- [A Semantic Construction Profile for Exact Boolean Formula Complexity](papers/semantic-construction-profile/note.pdf)

Code is MIT licensed. Original papers, reports, and figures are CC BY 4.0;
see [`LICENSE-CONTENT.md`](LICENSE-CONTENT.md).
