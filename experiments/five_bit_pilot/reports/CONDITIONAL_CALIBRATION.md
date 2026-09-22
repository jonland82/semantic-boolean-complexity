# Prospective conditional calibration at five inputs

The protocol was frozen before exact cost-11 enumeration. It trains
separate conditional lower/upper error models through cost 9, uses cost
10 only for conformal residual correction, and evaluates cost 11 once.
K-hat remains the interval center; all uncertainty features are semantic
and observable without knowing exact K.

| language   |   train_functions |   calibration_functions |   prospective_test_functions |   coverage |   mean_lower_allowance |   mean_upper_allowance |   mean_learned_width |   mean_classical_width |   mean_relative_shrinkage |   misses_below |   misses_above |
|:-----------|------------------:|------------------------:|-----------------------------:|-----------:|-----------------------:|-----------------------:|---------------------:|-----------------------:|--------------------------:|---------------:|---------------:|
| NAND       |               340 |                      40 |                           40 |      1     |                 3.1858 |                 1.1956 |               4.3814 |                83.083  |                    0.9466 |              0 |              0 |
| NOR        |               340 |                      40 |                           40 |      1     |                 3.9115 |                 1.1745 |               5.086  |                81.2454 |                    0.9375 |              0 |              0 |
| AND_OR_NOT |               350 |                      40 |                           40 |      0.925 |                 2.6813 |                 1.3771 |               4.0584 |                14.5691 |                    0.7191 |              0 |              3 |

Protocol SHA-256: `1aa93c5f1b3b073fe01754442814fc85879c8ed8361e00e01ffaf111c1cfc28d`

## Reproduce

```powershell
python experiments/five_bit_pilot/scripts/analyze_conditional_calibration.py
```
