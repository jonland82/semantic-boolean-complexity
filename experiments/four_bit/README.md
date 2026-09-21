# Complete four-input experiment

This experiment covers all 65,536 four-input Boolean functions. It computes
exact minimum formula sizes in NAND, NOR, and AND/OR/NOT, derives syntax-free
truth-table descriptors, and evaluates prediction under function, input-orbit,
and exact-descriptor-class holdouts.

Run the complete exact synthesis and original validation from the repository
root:

```powershell
python experiments/four_bit/scripts/run.py
```

The subsequent analyses consume its generated artifacts:

```powershell
python experiments/four_bit/scripts/analyze_gap.py
python experiments/four_bit/scripts/analyze_parity_intervention.py
python experiments/four_bit/scripts/analyze_feature_associations.py
python experiments/four_bit/scripts/analyze_feature_importance.py
python experiments/four_bit/scripts/analyze_complexity_sandwich.py
python experiments/four_bit/scripts/analyze_sandwich_scaling.py
```

Outputs are written to `artifacts/`, plots to `figures/`, and concise Markdown
summaries to `reports/`.

The sandwich analysis also computes exact minimum-cost prime DNF/CNF cover
constructions for AND/OR/NOT, verifies every lower and upper inequality over
the complete universe, and writes all equality cases to
`artifacts/complexity_sandwich_contacts.csv`.

Run its regression and artifact-consistency checks from the repository root:

```powershell
python -m unittest discover -s tests -v
```

The synthesis is exact by induction on gate cost. In a minimum formula, every
proper subformula must also be minimum for the function it computes; otherwise
replacing it would shorten the whole formula. The dynamic program therefore
retains only functions first reached at each cost and combines those minimum
layers. It reproduces the checked-in three-input minima and verifies NAND/NOR
duality over the complete four-input universe.
