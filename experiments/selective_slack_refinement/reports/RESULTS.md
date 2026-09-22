# Selective residual-slack refinement

The teacher is trained through cost 12. Exact cost-13 results set separate
low/medium/high safety corrections, each enlarged by the previously observed
cross-layer maximum-residual drift. Reductions below 0.25 are suppressed.

Base protocol SHA-256: `0d96038e33a6c7a7a4a5051688df62e924fee96fb823167cd59cfc8d7823e750`

Frozen cost-14 protocol SHA-256: `6e1b8cbe4cc9bc72c8768de756595aadcd82874aaec6600c071da2542aba3087`

Cross-layer buffer: 0.260819.

## Cost-13 calibration

| model   | cohort           | risk_stratum   |   functions |   offset |   coverage |   mean_reduction |   functions_improved |
|:--------|:-----------------|:---------------|------------:|---------:|-----------:|-----------------:|---------------------:|
| teacher | targeted         | high           |          46 |   2.3607 |          1 |           1.0982 |                   46 |
| teacher | targeted         | medium         |          34 |   2.1129 |          1 |           0.6562 |                   34 |
| teacher | random_reference | low            |          18 |   1.5668 |          1 |           0.0649 |                    4 |
| teacher | random_reference | medium         |          20 |   2.1129 |          1 |           0.2216 |                    8 |
| teacher | random_reference | high           |           2 |   2.3607 |          1 |           0.8632 |                    2 |
| student | targeted         | high           |           2 |   0.5986 |          1 |           2.7392 |                    2 |
| student | targeted         | medium         |          47 |   1.8379 |          1 |           0.7197 |                   47 |
| student | targeted         | low            |          31 |   1.9534 |          1 |           0      |                    0 |
| student | random_reference | low            |          37 |   1.9534 |          1 |           0      |                    0 |
| student | random_reference | medium         |           3 |   1.8379 |          1 |           0.7054 |                    3 |

## Exhaustive four-input audit

| model   |   functions |   coverage |   mean_reduction |   functions_improved |   low_offset |   medium_offset |   high_offset |
|:--------|------------:|-----------:|-----------------:|---------------------:|-------------:|----------------:|--------------:|
| teacher |       65536 |          1 |           0.2124 |                17391 |       1.1832 |          1.1295 |        0.6647 |
| student |       65536 |          1 |           0.0245 |                 2078 |       1.6926 |          1.5771 |        2.4489 |

## Frozen next test

The model hashes, feature order, stratum thresholds, offsets, abstention
threshold, sampling seed, cohort sizes, and success criteria are stored in
`../artifacts/frozen_cost14_protocol.json`.
