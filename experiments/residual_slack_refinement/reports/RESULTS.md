# Residual-slack refinement below certified constructions

This model predicts only the slack remaining below the certified
restriction/factor upper bound. Costs through 11 are development data;
cost 12 sets the one-sided safety correction; cost 13 remains untouched.

Base protocol SHA-256: `59c560728a2e7914ac666ad27dfc5a4cd058e17d2a0084e2684ed9f295a52efd`

Frozen cost-13 protocol SHA-256: `7114b96ad029d86a28a8fe5dd0e27a4bbc2b3c0d5fc14ce117106205b1135127`

## Model selection

| model                   |   selection_functions |   selection_offset |   selection_coverage |   selection_mean_reduction |   selection_functions_improved |
|:------------------------|----------------------:|-------------------:|---------------------:|---------------------------:|-------------------------------:|
| squared_boosting        |                 13203 |             1.4051 |                    1 |                       0.14 |                           3668 |
| lower_quantile_boosting |                 13203 |             0      |                    1 |                       0    |                              0 |

Selected teacher: `squared_boosting`.

## Cost-12 safety calibration

| model   | cohort                   |   functions |   calibration_offset |   coverage |   mean_construction_upper |   mean_refined_upper |   mean_reduction_beyond_construction |   functions_improved |
|:--------|:-------------------------|------------:|---------------------:|-----------:|--------------------------:|---------------------:|-------------------------------------:|---------------------:|
| teacher | cost_12_upper_tail       |          80 |               1.9    |          1 |                   15.7375 |              15.4849 |                               0.2526 |                   57 |
| teacher | cost_12_random_reference |          40 |               1.9    |          1 |                   15      |              14.7079 |                               0.2921 |                   21 |
| student | cost_12_upper_tail       |          80 |               1.7046 |          1 |                   15.7375 |              15.6457 |                               0.0918 |                   12 |
| student | cost_12_random_reference |          40 |               1.7046 |          1 |                   15      |              14.8948 |                               0.1052 |                    9 |

Coverage is 100% by construction because the maximum dangerous cost-12
residual is the frozen correction. The meaningful quantities are how much
positive reduction survives and whether it appears in both cohorts.

## Exhaustive four-input counterexample audit

| model   |   functions |   audit_offset |   coverage |   mean_reduction_beyond_construction |   functions_improved |   maximum_reduction |
|:--------|------------:|---------------:|-----------:|-------------------------------------:|---------------------:|--------------------:|
| teacher |       65536 |         1.3299 |          1 |                               0.1626 |                20540 |              4.5115 |
| student |       65536 |         2.4513 |          1 |                               0.0058 |                  764 |              0.8889 |

The maximum dangerous residual over the complete universe gives each row
a 100% finite-domain certificate. Known five-input raw counterexamples are
saved separately and were included before the cost-13 freeze.

## Frozen prospective test

The serialized teacher, feature order, cost-12 correction, selection seed,
cohort sizes, and success criteria are recorded in
`../artifacts/frozen_cost13_protocol.json`. Any cost-13 evaluation must verify
the recorded artifact hashes before enumeration.

## Reproduce

```powershell
python experiments/residual_slack_refinement/scripts/run_local_refinement.py
```
