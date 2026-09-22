# ML-assisted AON upper-bound slack discovery

The model predicts removable slack in the existing analytic upper bound.
A maximum one-sided residual from five-input cost 10 is subtracted from
the prediction before any reduction is made. Costs 11 and 12 are evaluation
only; the paper and previous experimental artifacts are unchanged.

Protocol SHA-256: `1e82cc96fcc0e0f53e979a41f89ffc93133592cfcfd648f33279c11c06b7b7c2`

Training functions: 52,635; calibration functions: 40.

## Five-input validation

| model        | cohort                   |   functions |   coverage |   misses |   mean_existing_upper |   mean_construction_upper |   mean_learned_upper |   mean_combined_upper |   mean_reduction |   functions_improved |   functions_beyond_construction |   mean_reduction_beyond_construction |   maximum_violation |
|:-------------|:-------------------------|------------:|-----------:|---------:|----------------------:|--------------------------:|---------------------:|----------------------:|-----------------:|---------------------:|--------------------------------:|-------------------------------------:|--------------------:|
| boosting     | cost_11                  |          40 |      0.975 |        1 |               18.2    |                   13.675  |              14.2276 |               13.4041 |           3.9724 |                   40 |                              15 |                               0.2709 |              0.4301 |
| boosting     | cost_12_upper_tail       |          80 |      1     |        0 |               24.6625 |                   15.7375 |              20.3764 |               15.7375 |           4.2861 |                   80 |                               0 |                               0      |              0      |
| boosting     | cost_12_random_reference |          40 |      1     |        0 |               19.95   |                   15      |              15.9183 |               14.7064 |           4.0317 |                   40 |                              13 |                               0.2936 |              0      |
| shallow_tree | cost_11                  |          40 |      0.95  |        2 |               18.2    |                   13.675  |              13.8729 |               13.3478 |           4.3271 |                   40 |                              18 |                               0.3272 |              1      |
| shallow_tree | cost_12_upper_tail       |          80 |      1     |        0 |               24.6625 |                   15.7375 |              19.909  |               15.7375 |           4.7535 |                   80 |                               0 |                               0      |              0      |
| shallow_tree | cost_12_random_reference |          40 |      0.975 |        1 |               19.95   |                   15      |              15.4319 |               14.618  |           4.5181 |                   40 |                              15 |                               0.382  |              0.7606 |
| ridge        | cost_11                  |          40 |      0.975 |        1 |               18.2    |                   13.675  |              14.339  |               13.622  |           3.861  |                   40 |                               7 |                               0.053  |              0.0619 |
| ridge        | cost_12_upper_tail       |          80 |      1     |        0 |               24.6625 |                   15.7375 |              16.4825 |               15.7063 |           8.18   |                   80 |                              13 |                               0.0312 |              0      |
| ridge        | cost_12_random_reference |          40 |      1     |        0 |               19.95   |                   15      |              15.5122 |               14.8891 |           4.4378 |                   40 |                              13 |                               0.1109 |              0      |

Coverage here is prospective or held-out empirical coverage, not a universal
guarantee. A miss means the model claimed more removable slack than existed.

The combined endpoint is the minimum of the learned upper bound and the
certified restriction/factor construction. `functions_beyond_construction`
counts cases where ML supplies a further reduction; its coverage remains
the learned bound's empirical coverage.

Boosting is the strongest robust candidate here: it covers both cost-12
cohorts completely and covers 39/40 cost-11 functions. Combined with the
certified construction, it improves a further 15/40 cost-11 functions and
13/40 cost-12 random-reference functions, but none of the selected cost-12
upper-tail functions. The interpretable tree is slightly more aggressive
and correspondingly has more misses.

## Exhaustive four-input audit

| model        |   functions |   audit_offset |   coverage |   mean_reduction |   functions_improved |   maximum_reduction |   combined_coverage |   mean_combined_reduction |   functions_beyond_construction |   mean_reduction_beyond_construction |
|:-------------|------------:|---------------:|-----------:|-----------------:|---------------------:|--------------------:|--------------------:|--------------------------:|--------------------------------:|-------------------------------------:|
| boosting     |       65536 |         1.2786 |          1 |           1.0629 |                57176 |              4.9213 |                   1 |                    1.3048 |                           22408 |                               0.1829 |
| shallow_tree |       65536 |         2.4057 |          1 |           0.2774 |                26488 |              2.9966 |                   1 |                    1.1275 |                             574 |                               0.0057 |
| ridge        |       65536 |         1.8526 |          1 |           0.5954 |                45221 |              4.3499 |                   1 |                    1.1384 |                            5332 |                               0.0165 |

The audit offset is the worst dangerous residual across all 65,536 four-input
functions. These audited bounds therefore have a complete finite-domain
certificate even though the underlying slack predictions are learned.

Because the exhaustively audited learned bound and the construction bound
are both valid on four inputs, their combined minimum is also a certified
finite-domain bound. `mean_reduction_beyond_construction` isolates the part
of that improvement attributable only to the learned correction.

For boosting, the combined bound remains valid on all 65,536 functions,
reduces the old upper endpoint by 1.3048 gates on average, and beats the
construction-only endpoint on 22,408 functions. The ML-only increment over
construction averages 0.1829 gates across the complete universe.

## Interpretable candidate

The complete depth-5 regression-tree rule list is stored in
`../artifacts/shallow_tree_rules.txt`. It is a discovery artifact rather
than a proved dimension-general inequality.

## Reproduce

```powershell
python experiments/slack_bound_discovery/scripts/run_slack_discovery.py
```
