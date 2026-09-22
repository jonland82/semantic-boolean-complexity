# Local selection-aware calibration study

This is a retrospective diagnostic. Cost 14 has already been observed,
so these results cannot validate a repaired rule prospectively.

## Design

A 90th-quantile model predicts the dangerous residual `score - true headroom`
from cost 12. Cost 13 supplies a maximum one-sided additive correction.
The repaired reduction is applied unchanged to cost 14. We compare score-only,
construction-aware, and full restriction-witness feature sets; each correction
is calibrated either on all cost-13 cases or only the model-selected cohort.
The original 1,000-candidate pools were not retained, so full top-80 replay is
not available locally.

A maximum-residual correction based on 80 selected calibration cases has
only `80 / (80 + 80) = 50%` probability of exceeding the maximum of a
future 80-case batch under ideal exchangeability. Reaching 99% by this
rank argument requires at least 7,920 selected calibration cases.

## Cost-14 diagnostic

| policy                            |   coverage |   misses |   maximum_violation |   mean_reduction |   fraction_of_baseline_gain |   improved_fraction |
|:----------------------------------|-----------:|---------:|--------------------:|-----------------:|----------------------------:|--------------------:|
| frozen_baseline                   |     0.95   |        6 |              0.4007 |           0.8795 |                      1      |              0.825  |
| frozen_unit_reduction_cap         |     1      |        0 |              0      |           0.7483 |                      0.8509 |              0.825  |
| frozen_abstain_endpoint_16        |     1      |        0 |              0      |           0.7536 |                      0.8569 |              0.6833 |
| score_only_all                    |     0.925  |        9 |              0.7321 |           1.0324 |                      1.1739 |              0.7833 |
| score_only_selected               |     0.925  |        9 |              0.7321 |           1.0324 |                      1.1739 |              0.7833 |
| construction_aware_all            |     0.675  |       39 |              0.8299 |           1.9114 |                      2.1733 |              1      |
| construction_aware_selected       |     0.675  |       39 |              0.8299 |           1.9114 |                      2.1733 |              1      |
| full_restriction_witness_all      |     0.6833 |       38 |              1.075  |           1.922  |                      2.1854 |              0.9833 |
| full_restriction_witness_selected |     0.6833 |       38 |              1.075  |           1.922  |                      2.1854 |              0.9833 |

## Historical cost-12 to cost-13 transport

| policy                            |   cost12_margin |   cost13_coverage |   cost13_misses |   cost13_mean_reduction |   cost13_maximum_violation |
|:----------------------------------|----------------:|------------------:|----------------:|------------------------:|---------------------------:|
| score_only_all                    |          0.8553 |            0.9167 |              10 |                  1.1049 |                     0.7727 |
| score_only_selected               |          0.308  |            0.8583 |              17 |                  1.6816 |                     1.3201 |
| construction_aware_all            |          0.5953 |            0.675  |              39 |                  2.1774 |                     0.7981 |
| construction_aware_selected       |          0.1358 |            0.3833 |              74 |                  2.5336 |                     1.2576 |
| full_restriction_witness_all      |          0.4333 |            0.525  |              57 |                  2.3424 |                     1.0019 |
| full_restriction_witness_selected |          0.1527 |            0.3583 |              77 |                  2.558  |                     1.2825 |

## Interpretation

The strongest diagnostic policy is `frozen_abstain_endpoint_16`: it has
0 misses, 0.7536 mean reduction, and
improves 68.3% of the 120 cases.
The learned dangerous-residual variants transport worse than the frozen
baseline. The only zero-miss local variants are conservative boundary
guards: capping any attempted refinement at one gate, or abstaining at
construction endpoint 16. The unit cap retains most of the original gain
and is the more general hypothesis; endpoint-16 abstention is tied to this
observed layer.
Because the six failures motivated this study, the result is hypothesis
generation only. A repaired policy must be chosen without reference to a
future labeled cohort, frozen, and then tested on new exact data.
