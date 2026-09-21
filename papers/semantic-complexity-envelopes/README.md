# Semantic complexity envelopes paper

This four-page note separates dimension-independent semantic bounds from the
sharp empirical constants verified on every three- and four-input Boolean
function.

Regenerate the analyses from the repository root:

```powershell
python experiments/four_bit/scripts/analyze_complexity_sandwich.py
python experiments/four_bit/scripts/analyze_sandwich_scaling.py
```

Compile from this directory:

```powershell
pdflatex -interaction=nonstopmode semantic-complexity-envelopes.tex
pdflatex -interaction=nonstopmode semantic-complexity-envelopes.tex
```
