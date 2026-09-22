# ML-assisted upper-bound slack discovery

This experiment trains models to estimate how much slack can be removed from
the existing AND/OR/NOT analytic upper bound. It uses one-sided calibration:
only overestimating removable slack can invalidate the learned upper bound.

The experiment is separate from the papers and from prior generated artifacts.
Run it locally from the repository root:

```powershell
python experiments/slack_bound_discovery/scripts/run_slack_discovery.py
```

Results are written to `reports/RESULTS.md`, with model summaries and evaluated
rows in `artifacts/`.
