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
