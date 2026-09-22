# Selective residual-slack refinement

This experiment tightens the certified AND/OR/NOT construction bound only
when a learned residual-slack model clears a risk-stratified safety correction
and a minimum useful-reduction threshold. Cost 13 calibrates the policy; cost
14 is reserved for one frozen prospective evaluation.

```powershell
python experiments/selective_slack_refinement/scripts/run_local_selective.py
python experiments/selective_slack_refinement/scripts/run_cost14_prospective.py
```

The scripts do not modify paper sources or previous experimental artifacts.

The completed frozen cost-14 evaluation is documented in
[`reports/COST14_RESULTS.md`](reports/COST14_RESULTS.md).
The post-hoc structural investigation of its six misses is documented in
[`reports/COST14_FAILURE_ANALYSIS.md`](reports/COST14_FAILURE_ANALYSIS.md).

```powershell
python experiments/selective_slack_refinement/scripts/analyze_cost14_failures.py
```

The retrospective local study of selection-aware, one-sided calibration is
documented in
[`reports/LOCAL_SELECTION_AWARE_RESULTS.md`](reports/LOCAL_SELECTION_AWARE_RESULTS.md).
It uses cost 12 for fitting, cost 13 for calibration, and the already-observed
cost 14 sample only as a diagnostic:

```powershell
python experiments/selective_slack_refinement/scripts/run_local_selection_aware.py
```
