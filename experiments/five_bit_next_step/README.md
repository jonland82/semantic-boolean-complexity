# Five-input AON upper-tail challenge

This is the next experiment after the learned semantic-envelope study. It is
kept separate from the paper and from the frozen cost-11 artifacts.

The frozen protocol targets exact cost 12, oversamples functions with
large prime-cover slack, and retains a seeded random reference group. It
compares the existing conditional envelope with a structure-augmented version
and audits whether restriction or disjoint-factor constructions improve the
certified upper endpoint.

Run from the repository root:

```powershell
python experiments/five_bit_next_step/scripts/run_aon_upper_tail.py
```

The run is CPU/RAM-bound and performs exhaustive sparse enumeration through
cost 12 to certify that every sampled target first appears at that cost.
Generated tables are written to `artifacts/`; the narrative result is written
to `reports/RESULTS.md`.
