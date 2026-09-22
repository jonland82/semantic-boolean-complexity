# Frozen prospective cost-14 selective result

Frozen protocol SHA-256: `6e1b8cbe4cc9bc72c8768de756595aadcd82874aaec6600c071da2542aba3087`

Exact cost-14 layer: 89,637,411 functions; exhaustive enumeration
through cost 14: 173,980,568 functions in
14258.7 seconds.

| cohort           |   functions |   coverage |   misses |   mean_construction_upper |   mean_selective_upper |   mean_reduction_beyond_construction |   functions_improved |   maximum_violation |
|:-----------------|------------:|-----------:|---------:|--------------------------:|-----------------------:|-------------------------------------:|---------------------:|--------------------:|
| targeted         |          80 |      0.925 |        6 |                   17.7625 |                16.5881 |                               1.1744 |                   80 |              0.4007 |
| random_reference |          40 |      1     |        0 |                   17.15   |                16.8605 |                               0.2895 |                   19 |              0      |

Overall coverage: 0.9500.
Maximum violation: 0.4007.
Mean reduction beyond construction: 0.8795.
Improved fraction: 0.8250.
Passed frozen success criteria: `False`.
