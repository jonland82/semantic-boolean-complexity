# Local five-input feasibility pilot

This pilot performs sparse exact formula enumeration through eleven gates for
NAND, NOR, and AND/OR/NOT. It samples each exact-cost layer, computes the frozen
16-coordinate semantic profile using degree bins `0, 1, 2, 3, 4+`, and applies
a model trained on four-input functions against the dimension-independent
decision-tree envelope, tightened by exact prime covers for AND/OR/NOT.

Run from the repository root:

```powershell
python experiments/five_bit_pilot/scripts/run_local_pilot.py
python experiments/five_bit_pilot/scripts/analyze_dimension_calibration.py
python experiments/five_bit_pilot/scripts/analyze_conditional_calibration.py
```

The sampled functions are limited to exact cost at most eleven.
They establish local feasibility and expose dimension shift; they are not a
representative five-input benchmark.
