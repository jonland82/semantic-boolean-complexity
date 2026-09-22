# Five-input dimension calibration

The four-input model and point predictions remain frozen. Exact five-input
samples from cost layers 0--8 calibrate only the two residual allowances;
the previously unseen exact cost-10 layer is held out for evaluation.

| language   | method                    |   calibration_functions |   test_functions |   lower_allowance |   upper_allowance |   coverage |   mean_learned_width |   mean_relative_shrinkage |   misses_below |   misses_above |
|:-----------|:--------------------------|------------------------:|-----------------:|------------------:|------------------:|-----------:|---------------------:|--------------------------:|---------------:|---------------:|
| NAND       | n4_zero_shot              |                       0 |               40 |            1.6535 |            1.531  |      0.925 |               3.1845 |                    0.957  |              2 |              1 |
| NAND       | n5_cost_0_8_calibrated    |                     300 |               40 |            2.6543 |            0.822  |      0.775 |               3.4763 |                    0.953  |              1 |              8 |
| NAND       | n5_affine_khat_adaptation |                      40 |               40 |            1.0411 |            2.3143 |      0.85  |               3.3554 |                    0.9546 |              1 |              5 |
| NOR        | n4_zero_shot              |                       0 |               40 |            1.8546 |            1.781  |      0.8   |               3.6356 |                    0.9522 |              8 |              0 |
| NOR        | n5_cost_0_8_calibrated    |                     300 |               40 |            3.4849 |            0.3613 |      0.85  |               3.8462 |                    0.9494 |              1 |              5 |
| NOR        | n5_affine_khat_adaptation |                      40 |               40 |            2.9602 |            2.2886 |      0.95  |               5.2488 |                    0.931  |              0 |              2 |
| AND_OR_NOT | n4_zero_shot              |                       0 |               40 |            1.3956 |            1.0443 |      0.725 |               2.4399 |                    0.8089 |              2 |              9 |
| AND_OR_NOT | n5_cost_0_8_calibrated    |                     310 |               40 |            1.6398 |            0.9561 |      0.675 |               2.5959 |                    0.7967 |              2 |             11 |
| AND_OR_NOT | n5_affine_khat_adaptation |                      40 |               40 |            2.0677 |            1.3069 |      0.7   |               3.3745 |                    0.7357 |              0 |             12 |

This is a small, cost-layer-balanced feasibility test. It demonstrates
whether a modest amount of target-dimension calibration repairs zero-shot
coverage. The affine variant fits a correction of K-hat on costs 0--7,
uses cost 8 only for calibration, ignores cost 9, and tests on cost 10.
This
cost-layer-balanced pilot is not representative of arbitrary functions.

## Reproduce

```powershell
python experiments/five_bit_pilot/scripts/analyze_dimension_calibration.py
```
