# Post-hoc structural audit of the six cost-14 misses

This analysis is exploratory: cost-14 labels are now observed. It
diagnoses the frozen failure but does not repair or reclassify the
prospective result.

## Central finding

All six misses lie in one sharply defined headroom regime. Their
certified construction endpoint is 16 while exact `K+1` is 15, so
only one gate is actually removable. The high-stratum correction was
2.360706; every model score above 3.360706 therefore
proposed a reduction larger than the available one-gate slack.

| truth table   | covered   |   ones |   predicted slack |   reduction |   violation |   tree leaves | best cofactor costs   |
|:--------------|:----------|-------:|------------------:|------------:|------------:|--------------:|:----------------------|
| 0x58760a6e    | False     |     15 |           3.76136 |    1.40066  |    0.400656 |            15 | 4+7                   |
| 0xb0aae8c0    | False     |     13 |           3.62402 |    1.26331  |    0.263312 |            14 | 5+6                   |
| 0xf15c00a8    | False     |     12 |           3.58834 |    1.22764  |    0.227635 |            13 | 7+4                   |
| 0x05e402e8    | False     |     11 |           3.58616 |    1.22545  |    0.225451 |            13 | 8+3                   |
| 0x0a172737    | False     |     15 |           3.52706 |    1.16636  |    0.166359 |            12 | 5+6                   |
| 0xcdd46774    | False     |     18 |           3.38032 |    1.01962  |    0.019617 |            13 | 5+6                   |
| 0x7676ec28    | True      |     17 |           3.32383 |    0.963129 |    0        |            12 | 4+7                   |
| 0xf8ba1102    | True      |     13 |           3.32097 |    0.960269 |    0        |            13 | 8+3                   |
| 0xfb8aba32    | True      |     18 |           3.28088 |    0.92017  |    0        |            13 | 6+5                   |
| 0x21252111    | True      |      9 |           2.99835 |    0.885405 |    0        |            12 | 7+4                   |

The same construction-16 regime contains four covered targeted
controls. Sorted by model score, the six misses are exactly the six
highest-scored cases; the four controls are the four lowest. Thus the
failure is a tail-ranking effect inside a low-headroom subgroup, not
six unrelated numerical accidents.

## Calibration-support shift

At cost 13, only 1/46 targeted high-stratum
functions had one gate of construction slack. At cost 14 the count
was 9/78, a 5.31x increase
in prevalence. The maximum dangerous residual moved from
2.099887 to 2.761361. After the frozen cross-layer
buffer, the correction still fell short by 0.400656,
which is exactly the largest observed violation.

|   cost |   functions |   one_slack_functions |   one_slack_fraction |   mean_prediction |   maximum_dangerous_residual |
|-------:|------------:|----------------------:|---------------------:|------------------:|-----------------------------:|
|     13 |          46 |                     1 |             0.021739 |           3.45886 |                      2.09989 |
|     14 |          78 |                     9 |             0.115385 |           3.54257 |                      2.76136 |

The cost-13 calibration set contained only one direct analogue: a
targeted, high-stratum, one-slack case. Its prediction was 3.099887.
The cost-14 one-slack predictions extended to 3.761361. A maximum
residual correction cannot protect a tail that was scarcely represented
when the correction was frozen.

## Structural checks

The ten construction-16 cases occupy 10 distinct NPN
equivalence classes, so the misses are not duplicates under input
permutation, input negation, or output negation. Every case has one
unique best restriction variable, and the best four-input cofactor
cost pairs span 3+8, 4+7, 5+6. Covered controls exhibit the
same cofactor-cost patterns. None is canalizing or disjoint-variable
AND/OR factorable. The shared structure is therefore broader: a
restriction construction nearly closes the problem even though the
remaining semantic profile still looks difficult to the model.

A one-feature-at-a-time local perturbation identifies the coordinates
that most strongly distinguish the six high scores from the four
matched controls:

| feature                   |   mean_push_misses |   mean_push_covered_controls |   miss_minus_control_push |
|:--------------------------|-------------------:|-----------------------------:|--------------------------:|
| decision_tree_leaf_count  |           0.098698 |                    -0.111691 |                  0.21039  |
| anf_terms_degree_1        |           0.359388 |                     0.190027 |                  0.16936  |
| anf_terms_degree_2        |           0.146306 |                     0.05251  |                  0.093796 |
| fourier_negative_degree_2 |           0.024796 |                    -0.031762 |                  0.056558 |

Decision-tree leaf count and low-degree ANF support are the leading
upward score drivers in this matched group. Restriction gain has much
smaller contrast. This diagnostic is not causal---the coordinates are
correlated---but it supports a coherent interpretation: the teacher
continues to see global semantic difficulty after the restriction
construction has already reduced the true remaining slack to one gate.

## Implication

The failure is primarily a calibration-and-interaction failure, not a
lack of semantic signal. The next model should expose the complete
restriction witness profile (cofactor costs, imbalance, and gap to the
second-best restriction), and the safety correction should be calibrated
on the entire top-k selection pipeline. A one-sided quantile or dangerous-
residual model is better aligned with the goal than squared-error slack
prediction. Any rule designed from these six cases is post-hoc and must
be frozen before a new replication sample.
