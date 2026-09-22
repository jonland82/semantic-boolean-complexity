# Learned semantic complexity envelope

The compact sandwich model predicts normalized position between the
Khrapchenko lower endpoint and the decision-tree/prime-cover upper endpoint.
Every point prediction below is out-of-fold over complete 16-coordinate
semantic-profile classes.

## Interval results

| language   | method                               | allowance_units            |   nominal_coverage |   mean_lower_allowance |   mean_upper_allowance |   maximum_lower_allowance |   maximum_upper_allowance |   function_coverage |   mean_classical_width |   mean_learned_width |   mean_relative_shrinkage |   median_relative_shrinkage |   fraction_strictly_narrower |   mean_classical_integer_candidates |   mean_learned_integer_candidates |   mean_integer_candidate_reduction |   class_coverage |   minimum_shrinkage_guarantee |
|:-----------|:-------------------------------------|:---------------------------|-------------------:|-----------------------:|-----------------------:|--------------------------:|--------------------------:|--------------------:|-----------------------:|---------------------:|--------------------------:|----------------------------:|-----------------------------:|------------------------------------:|----------------------------------:|-----------------------------------:|-----------------:|------------------------------:|
| NAND       | cross_fitted_exhaustive_max_residual | target_units               |               1    |                 4.2396 |                 3.9255 |                    4.2396 |                    3.9255 |              1      |                18.3788 |               8.1573 |                    0.5467 |                      0.561  |                       1      |                             18.3471 |                            8.1607 |                             0.5455 |           1      |                      nan      |
| NAND       | split_conformal_class_worst          | target_units               |               0.9  |                 1.3254 |                 1.2663 |                    1.4695 |                    1.3447 |              0.9183 |                18.3788 |               2.5903 |                    0.856  |                      0.8599 |                       1      |                             18.3471 |                            2.5888 |                             0.8557 |           0.9018 |                      nan      |
| NAND       | split_conformal_gap_normalized       | fraction_of_analytic_width |               0.9  |                 0.0721 |                 0.0706 |                    0.0786 |                    0.0781 |              0.9137 |                18.3788 |               2.6216 |                    0.8574 |                      0.8547 |                       1      |                             18.3471 |                            2.6217 |                             0.8571 |           0.9018 |                        0.8516 |
| NAND       | split_conformal_class_worst          | target_units               |               0.95 |                 1.6723 |                 1.5604 |                    1.8007 |                    1.7281 |              0.9631 |                18.3788 |               3.232  |                    0.8203 |                      0.8242 |                       1      |                             18.3471 |                            3.2274 |                             0.8202 |           0.9509 |                      nan      |
| NAND       | split_conformal_gap_normalized       | fraction_of_analytic_width |               0.95 |                 0.0903 |                 0.0858 |                    0.1028 |                    0.0969 |              0.9611 |                18.3788 |               3.2351 |                    0.824  |                      0.8223 |                       1      |                             18.3471 |                            3.2372 |                             0.8235 |           0.9502 |                        0.8135 |
| NAND       | split_conformal_class_worst          | target_units               |               0.99 |                 2.5328 |                 2.2581 |                    2.9792 |                    2.3729 |              0.9934 |                18.3788 |               4.7852 |                    0.734  |                      0.7412 |                       1      |                             18.3471 |                            4.7845 |                             0.7334 |           0.9899 |                      nan      |
| NAND       | split_conformal_gap_normalized       | fraction_of_analytic_width |               0.99 |                 0.1388 |                 0.1268 |                    0.1632 |                    0.139  |              0.9946 |                18.3788 |               4.8773 |                    0.7347 |                      0.7274 |                       1      |                             18.3471 |                            4.8877 |                             0.7335 |           0.9919 |                        0.7084 |
| NOR        | cross_fitted_exhaustive_max_residual | target_units               |               1    |                 3.6108 |                 4.1064 |                    3.6108 |                    4.1064 |              1      |                18.3788 |               7.7135 |                    0.5713 |                      0.5851 |                       1      |                             18.3471 |                            7.7016 |                             0.5709 |           1      |                      nan      |
| NOR        | split_conformal_class_worst          | target_units               |               0.9  |                 1.5442 |                 1.4987 |                    1.5724 |                    1.6083 |              0.923  |                18.3788 |               3.0426 |                    0.8308 |                      0.8334 |                       1      |                             18.3471 |                            3.0482 |                             0.8301 |           0.9038 |                      nan      |
| NOR        | split_conformal_gap_normalized       | fraction_of_analytic_width |               0.9  |                 0.0825 |                 0.0828 |                    0.085  |                    0.0871 |              0.9167 |                18.3788 |               3.0375 |                    0.8347 |                      0.8357 |                       1      |                             18.3471 |                            3.0403 |                             0.8341 |           0.9026 |                        0.8279 |
| NOR        | split_conformal_class_worst          | target_units               |               0.95 |                 1.9147 |                 1.8297 |                    2.0497 |                    1.9321 |              0.9663 |                18.3788 |               3.7431 |                    0.7919 |                      0.7968 |                       1      |                             18.3471 |                            3.7376 |                             0.7918 |           0.9527 |                      nan      |
| NOR        | split_conformal_gap_normalized       | fraction_of_analytic_width |               0.95 |                 0.1024 |                 0.0991 |                    0.1048 |                    0.1052 |              0.9621 |                18.3788 |               3.7043 |                    0.7985 |                      0.7971 |                       1      |                             18.3471 |                            3.6994 |                             0.7982 |           0.9512 |                        0.79   |
| NOR        | split_conformal_class_worst          | target_units               |               0.99 |                 2.8134 |                 2.6077 |                    3.1625 |                    2.9633 |              0.997  |                18.3788 |               5.4184 |                    0.6988 |                      0.7062 |                       1      |                             18.3471 |                            5.4116 |                             0.6985 |           0.9944 |                      nan      |
| NOR        | split_conformal_gap_normalized       | fraction_of_analytic_width |               0.99 |                 0.1563 |                 0.1431 |                    0.1694 |                    0.1617 |              0.9967 |                18.3788 |               5.5024 |                    0.7006 |                      0.6959 |                       1      |                             18.3471 |                            5.5084 |                             0.6996 |           0.9947 |                        0.6922 |
| AND_OR_NOT | cross_fitted_exhaustive_max_residual | target_units               |               1    |                 2.7429 |                 2.0121 |                    2.7429 |                    2.0121 |              1      |                 9.0669 |               4.5528 |                    0.4786 |                      0.502  |                       0.9936 |                              9.148  |                            4.6521 |                             0.4717 |           1      |                      nan      |
| AND_OR_NOT | split_conformal_class_worst          | target_units               |               0.9  |                 1.0758 |                 0.9204 |                    1.2354 |                    0.9601 |              0.9191 |                 9.0669 |               1.9792 |                    0.771  |                      0.7858 |                       0.9997 |                              9.148  |                            1.9934 |                             0.7712 |           0.9056 |                      nan      |
| AND_OR_NOT | split_conformal_gap_normalized       | fraction_of_analytic_width |               0.9  |                 0.1176 |                 0.1066 |                    0.135  |                    0.1112 |              0.9125 |                 9.0669 |               2.0244 |                    0.7771 |                      0.7774 |                       0.9999 |                              9.148  |                            2.0287 |                             0.7777 |           0.9008 |                        0.759  |
| AND_OR_NOT | split_conformal_class_worst          | target_units               |               0.95 |                 1.3676 |                 1.1268 |                    1.6606 |                    1.2043 |              0.9636 |                 9.0669 |               2.4644 |                    0.7153 |                      0.7321 |                       0.9995 |                              9.148  |                            2.4764 |                             0.7168 |           0.9524 |                      nan      |
| AND_OR_NOT | split_conformal_gap_normalized       | fraction_of_analytic_width |               0.95 |                 0.1445 |                 0.1285 |                    0.1664 |                    0.1363 |              0.9619 |                 9.0669 |               2.46   |                    0.7292 |                      0.7299 |                       0.9999 |                              9.148  |                            2.4804 |                             0.729  |           0.9537 |                        0.7076 |
| AND_OR_NOT | split_conformal_class_worst          | target_units               |               0.99 |                 1.9291 |                 1.5384 |                    2.2014 |                    1.6936 |              0.9924 |                 9.0669 |               3.3851 |                    0.6106 |                      0.6329 |                       0.9984 |                              9.148  |                            3.4211 |                             0.6098 |           0.9904 |                      nan      |
| AND_OR_NOT | split_conformal_gap_normalized       | fraction_of_analytic_width |               0.99 |                 0.2092 |                 0.1886 |                    0.2603 |                    0.2307 |              0.994  |                 9.0669 |               3.5301 |                    0.6118 |                      0.6079 |                       0.9999 |                              9.148  |                            3.5622 |                             0.6105 |           0.9922 |                        0.5626 |

The exhaustive row uses the largest observed one-sided residual of all
65,536 cross-fitted predictions. It is a finite audit of this fixed table,
not a dimension-independent theorem. The calibrated rows use a stricter
three-fold train, one-fold calibrate, one-fold test rotation. Calibration
scores take the worst residual inside each complete semantic class; the two
tails use a Bonferroni split of the stated joint error rate.

For any single split, if the held-out profile class is exchangeable with
the calibration classes, the finite-sample higher quantile gives at least
the nominal simultaneous class-uniform coverage by the union bound. The
five rotating splits reported here are an empirical aggregation of that
grouped split-conformal construction.

The gap-normalized rows divide each residual by the width of its analytic
envelope before calibration. Their allowances are therefore fractions,
not gate-count units. If the two fold-specific fractions sum to less than
one, the analytic-gap theorem guarantees that every noncollapsed interval
in that fold shrinks by at least one minus their sum, independently of
whether its target is covered.

Intervals are always intersected with the classical semantic envelope.
Coverage therefore cannot be worse than an un-intersected learned interval.

![Learned envelope shrinkage](../figures/learned_envelope_shrinkage.png)

## Largest cross-fitted point errors

| language   | truth_table   |   exact_k_plus_1 |   predicted_k_plus_1 |   signed_residual_exact_minus_prediction |   classical_lower_k_plus_1 |   classical_upper_k_plus_1 |
|:-----------|:--------------|-----------------:|---------------------:|-----------------------------------------:|---------------------------:|---------------------------:|
| AND_OR_NOT | 0xeac7        |                9 |               11.743 |                                   -2.743 |                      4.267 |                     14.556 |
| AND_OR_NOT | 0xead3        |                9 |               11.743 |                                   -2.743 |                      4.267 |                     14.556 |
| AND_OR_NOT | 0xebc5        |                9 |               11.743 |                                   -2.743 |                      4.267 |                     14.556 |
| AND_OR_NOT | 0xebd1        |                9 |               11.743 |                                   -2.743 |                      4.267 |                     14.556 |
| AND_OR_NOT | 0xeca7        |                9 |               11.743 |                                   -2.743 |                      4.267 |                     14.556 |
| NAND       | 0x6996        |               24 |               28.24  |                                   -4.24  |                     16     |                     39.75  |
| NAND       | 0x0000        |                6 |                2.074 |                                    3.926 |                      0     |                      6     |
| NAND       | 0x9669        |               24 |               27.646 |                                   -3.646 |                     16     |                     39.75  |
| NAND       | 0x6bde        |               16 |               19.451 |                                   -3.451 |                      7.273 |                     30.75  |
| NAND       | 0x6bf6        |               16 |               19.451 |                                   -3.451 |                      7.273 |                     30.75  |
| NOR        | 0xffff        |                6 |                1.894 |                                    4.106 |                      0     |                      6     |
| NOR        | 0x6996        |               24 |               27.611 |                                   -3.611 |                     16     |                     39.75  |
| NOR        | 0x6eef        |               16 |               12.478 |                                    3.522 |                      3     |                     21.75  |
| NOR        | 0x7afb        |               16 |               12.478 |                                    3.522 |                      3     |                     21.75  |
| NOR        | 0x7cfd        |               16 |               12.478 |                                    3.522 |                      3     |                     21.75  |

## Interpretation

A useful learned envelope must jointly achieve high held-out coverage and
material width reduction. Mean and median relative shrinkage are computed
per function against the original interval; already-collapsed classical
intervals receive shrinkage one by convention. Integer-candidate columns
also round the endpoints to the feasible integral values of `K + 1`.

The model supplies the interval center, while calibration or exhaustive
verification supplies the error allowance. Only the classical endpoints
remain dimension-independent mathematical bounds at this stage.

## Reproduce

```powershell
python experiments/four_bit/scripts/analyze_learned_envelope.py
```
