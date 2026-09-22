# Five-input AON upper-tail challenge

This experiment is separate from the learned-envelope paper and leaves
the prospective cost-11 result unchanged. The protocol was written and hashed
before exact cost-12 enumeration and selection did not use prediction error.

Protocol SHA-256: `939a1bd6d60da5adf3aebe24fe2d3ee215c78bfd1867248db8fa818535a84bf0`

Exact cost-12 layer: 22,092,002 functions; exhaustive enumeration
through cost 12: 37,723,540 functions in
1068.9 seconds.

## Envelope results

| cohort           | method              |   functions |   coverage |   misses_below |   misses_above |   mean_learned_width |   mean_relative_shrinkage |
|:-----------------|:--------------------|------------:|-----------:|---------------:|---------------:|---------------------:|--------------------------:|
| upper_tail       | baseline            |          80 |     0.9125 |              7 |              0 |               4.1745 |                    0.7932 |
| upper_tail       | structure_augmented |          80 |     0.8875 |              9 |              0 |               4.003  |                    0.8017 |
| random_reference | baseline            |          40 |     0.775  |              0 |              9 |               4.0734 |                    0.7434 |
| random_reference | structure_augmented |          40 |     0.775  |              0 |              9 |               3.9184 |                    0.753  |

The upper-tail cohort has the largest prime-cover slack in a seeded
1,000-function candidate pool. The random-reference cohort is sampled
from the remainder. These layer-balanced challenge results are diagnostic,
not a coverage theorem for arbitrary five-input functions.

The structure features make both cohorts slightly narrower, but they do
not improve coverage: targeted coverage falls from 91.25% to 88.75%,
while random-reference coverage remains 77.5%. The opposite miss directions
across cohorts show that prime-cover-slack selection induces a strong shift
in the frozen point prediction, not a uniformly harder upper residual.

## Certified construction results

| cohort           |   functions |   functions_improved |   restriction_improved |   factor_improved |   mean_existing_upper |   mean_construction_upper |   mean_certified_improvement |
|:-----------------|------------:|---------------------:|-----------------------:|------------------:|------------------------:|--------------------------:|-----------------------------:|
| upper_tail       |          80 |                   80 |                     80 |                 0 |                 24.6625 |                   15.7375 |                        8.925 |
| random_reference |          40 |                   39 |                     39 |                 0 |                 19.95   |                   15      |                        4.95  |

The construction endpoint is the minimum of the existing prime-cover/tree
bound, exact four-input cofactor restriction decompositions, and disjoint
AND/OR factorizations. Every reported improvement is a certified formula
upper bound, independent of learned-envelope coverage. In this sample all
improvements come from restriction decompositions; disjoint factorization
does not beat the existing endpoint.

## Reproduce

```powershell
python experiments/five_bit_next_step/scripts/run_aon_upper_tail.py
```
