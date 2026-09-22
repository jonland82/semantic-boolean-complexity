# Residual-slack refinement

This experiment asks whether machine learning can safely remove additional
slack after applying the certified restriction/factor construction bound.
It treats the observed cost-11 cases as development data, uses cost 12 only
for the one-sided safety correction, and emits a frozen cost-13 protocol.

Run the local discovery and freezing stage:

```powershell
python experiments/residual_slack_refinement/scripts/run_local_refinement.py
```

The prospective cost-13 runner is generated for a separate compute job:

```powershell
python experiments/residual_slack_refinement/scripts/run_cost13_prospective.py
```

The completed frozen evaluation is documented in
[`reports/COST13_RESULTS.md`](reports/COST13_RESULTS.md).

No paper sources are read or modified by either script.
