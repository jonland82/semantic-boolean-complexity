# Local five-input learned-envelope pilot

This is a feasibility run over sampled functions whose exact minimum
formula cost is at most eleven. Exactness follows from exhaustive sparse
enumeration of every function reachable at each gate cost through eleven.
It is strongly biased toward easy functions and is not a representative
five-input validation set.

| language   |   sampled_functions |   maximum_exact_cost |   coverage |   mean_exact_minus_prediction |   misses_below_interval |   misses_above_interval |   mean_classical_width |   mean_learned_width |   mean_relative_shrinkage |   enumerated_functions_through_max_cost |   enumeration_seconds |
|:-----------|--------------------:|---------------------:|-----------:|------------------------------:|------------------------:|------------------------:|-----------------------:|---------------------:|--------------------------:|----------------------------------------:|----------------------:|
| NAND       |                 420 |                   11 |     0.8452 |                       -0.6925 |                      62 |                       3 |                54.8872 |               3.1831 |                    0.9288 |                             4.43472e+06 |               21.0867 |
| NOR        |                 420 |                   11 |     0.8619 |                       -0.8753 |                      58 |                       0 |                54.9106 |               3.6342 |                    0.9193 |                             4.43472e+06 |               22.0899 |
| AND_OR_NOT |                 430 |                   11 |     0.8953 |                       -0.0531 |                      19 |                      26 |                 7.225  |               2.1007 |                    0.5633 |                             1.56315e+07 |              289.695  |

The predictor is refit on four-input data using the general decision-tree
upper bound, tightened by the exact prime-cover construction for
AND/OR/NOT. Degree coordinates use fixed bins 0, 1, 2, 3, and 4+.

Coverage here is diagnostic only. A decisive transfer experiment requires
exact targets beyond cost eleven, selected before observing errors.

## Reproduce

```powershell
python experiments/five_bit_pilot/scripts/run_local_pilot.py
```
