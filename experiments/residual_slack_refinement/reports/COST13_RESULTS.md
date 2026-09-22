# Frozen prospective cost-13 result

Frozen protocol SHA-256: `7114b96ad029d86a28a8fe5dd0e27a4bbc2b3c0d5fc14ce117106205b1135127`

Exact cost-13 layer: 46,619,617 functions; exhaustive enumeration
through cost 13: 84,343,157 functions in
3485.1 seconds.

| cohort           |   functions |   coverage |   misses |   mean_construction_upper |   mean_refined_upper |   mean_reduction_beyond_construction |   functions_improved |   maximum_violation |
|:-----------------|------------:|-----------:|---------:|--------------------------:|---------------------:|-------------------------------------:|---------------------:|--------------------:|
| targeted         |          80 |     0.9875 |        1 |                   16.7625 |              15.8749 |                               0.8876 |                   80 |              0.2608 |
| random_reference |          40 |     1      |        0 |                   15.925  |              15.8587 |                               0.0663 |                   10 |              0      |

Overall coverage: 0.9917.
Maximum violation: 0.2608.
Mean reduction beyond construction: 0.6139.
Passed frozen success criteria: `True`.
