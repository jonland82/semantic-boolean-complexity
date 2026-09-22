"""Compare learned-envelope shrinkage across semantic feature families."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPERIMENT = Path(__file__).resolve().parents[1]
ARTIFACTS = EXPERIMENT / "artifacts"
FIGURES = EXPERIMENT / "figures"
REPORTS = EXPERIMENT / "reports"


def load_learned_module():
    path = EXPERIMENT / "scripts" / "analyze_learned_envelope.py"
    spec = importlib.util.spec_from_file_location("learned_envelope_core", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_table(core):
    features = pd.read_csv(ARTIFACTS / "feature_table.csv")
    positions = pd.read_csv(ARTIFACTS / "complexity_sandwich_positions.csv")
    columns = [
        "truth_table",
        *[f"lower_{language}" for language in core.LANGUAGES],
        *[f"upper_{language}" for language in core.LANGUAGES],
        *[f"position_{language}" for language in core.LANGUAGES],
    ]
    return features.merge(positions[columns], on="truth_table", validate="one_to_one")


def main() -> None:
    core = load_learned_module()
    table = load_table(core)
    unique, _, groups, folds, fold_of_class = core.class_layout(table)
    compact_index = {name: index for index, name in enumerate(core.COMPACT)}
    feature_sets = {
        "fixed midpoint": [],
        "bracket coordinates": ["khrapchenko_product", "decision_tree_leaf_count"],
        "certificates": core.CERTIFICATES,
        "algebraic + phase": core.ALGEBRAIC_PHASE,
        "all corrections": core.CERTIFICATES + core.ALGEBRAIC_PHASE,
        "compact sandwich": core.COMPACT,
    }
    rows = []
    coverage = 0.95
    side_probability = 1 - (1 - coverage) / 2

    for language_index, language in enumerate(core.LANGUAGES):
        exact = table[f"target_{language}"].to_numpy(float) + 1
        lower = table[f"lower_{language}"].to_numpy(float)
        upper = table[f"upper_{language}"].to_numpy(float)
        width = upper - lower
        position = table[f"position_{language}"].to_numpy(float)
        weights, position_mean = core.class_means(position, groups, len(unique))

        for model_index, (model_name, names) in enumerate(feature_sets.items()):
            columns = [compact_index[name] for name in names]
            predicted_class_position = np.empty(len(unique), dtype=float)
            for fold, (train, test) in enumerate(folds):
                if not columns:
                    predicted_class_position[test] = 0.5
                else:
                    model = core.make_model(
                        language_index, fold, purpose=10 + model_index)
                    model.fit(unique[train][:, columns], position_mean[train],
                              sample_weight=weights[train])
                    predicted_class_position[test] = np.clip(
                        model.predict(unique[test][:, columns]), 0, 1)
            point_prediction = lower + predicted_class_position[groups] * width
            residual = exact - point_prediction
            point_r2 = 1 - float(np.sum(residual ** 2)) / float(
                np.sum((exact - np.mean(exact)) ** 2))
            point_rmse = float(np.sqrt(np.mean(residual ** 2)))

            learned_lower = np.empty(len(table), dtype=float)
            learned_upper = np.empty(len(table), dtype=float)
            allowances_minus = []
            allowances_plus = []
            for test_fold in range(core.FOLDS):
                calibration_fold = (test_fold + 1) % core.FOLDS
                test_classes = np.flatnonzero(fold_of_class == test_fold)
                calibration_classes = np.flatnonzero(
                    fold_of_class == calibration_fold)
                train_classes = np.flatnonzero(
                    (fold_of_class != test_fold)
                    & (fold_of_class != calibration_fold))
                if not columns:
                    nested_class_position = np.full(len(unique), 0.5)
                else:
                    model = core.make_model(
                        language_index, test_fold, purpose=20 + model_index)
                    model.fit(unique[train_classes][:, columns],
                              position_mean[train_classes],
                              sample_weight=weights[train_classes])
                    nested_class_position = np.clip(
                        model.predict(unique[:, columns]), 0, 1)
                nested_prediction = lower + nested_class_position[groups] * width
                scores_minus, scores_plus = core.class_worst_scores(
                    nested_prediction, exact, groups, calibration_classes)
                allowance_minus = core.conformal_higher_quantile(
                    scores_minus, side_probability)
                allowance_plus = core.conformal_higher_quantile(
                    scores_plus, side_probability)
                allowances_minus.append(allowance_minus)
                allowances_plus.append(allowance_plus)
                test_members = np.isin(groups, test_classes)
                learned_lower[test_members] = np.maximum(
                    lower[test_members],
                    nested_prediction[test_members] - allowance_minus)
                learned_upper[test_members] = np.minimum(
                    upper[test_members],
                    nested_prediction[test_members] + allowance_plus)

            metrics = core.interval_metrics(
                exact, lower, upper, learned_lower, learned_upper, groups)
            rows.append({
                "language": language,
                "model": model_name,
                "features": len(columns),
                "point_r2": point_r2,
                "point_rmse": point_rmse,
                "nominal_coverage": coverage,
                "mean_lower_allowance": float(np.mean(allowances_minus)),
                "mean_upper_allowance": float(np.mean(allowances_plus)),
                **metrics,
            })

    summary = pd.DataFrame(rows)
    summary.to_csv(ARTIFACTS / "learned_envelope_ablation_summary.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.2), sharey=True)
    for axis, language in zip(axes, core.LANGUAGES):
        subset = summary[summary.language == language]
        axis.barh(subset.model, 100 * subset.mean_integer_candidate_reduction,
                  color="#B98249")
        axis.set_title(core.LABELS[language])
        axis.set_xlabel("integer candidates removed (%)")
        axis.grid(axis="x", alpha=0.2)
    axes[0].set_ylabel("feature model")
    fig.suptitle("95% learned-envelope ablations", y=1.02)
    fig.tight_layout()
    for extension in ("png", "pdf", "svg"):
        fig.savefig(FIGURES / f"learned_envelope_ablations.{extension}",
                    dpi=220, bbox_inches="tight")
    plt.close(fig)

    display = summary[[
        "language", "model", "features", "point_r2", "point_rmse",
        "function_coverage", "class_coverage", "mean_relative_shrinkage",
        "mean_integer_candidate_reduction",
        "mean_learned_integer_candidates",
    ]].copy()
    for column in display.columns[3:]:
        display[column] = display[column].round(4)
    lines = [
        "# Learned-envelope feature ablations", "",
        "Each row uses the same compact-profile folds and nominal 95% grouped",
        "calibration protocol. The midpoint has no learned features.", "",
        display.to_markdown(index=False), "",
        "![Feature ablations](../figures/learned_envelope_ablations.png)", "",
        "The compact model should be judged against the midpoint by interval",
        "shrinkage at comparable held-out coverage, not only by point-prediction",
        "R-squared.", "", "## Reproduce", "", "```powershell",
        "python experiments/four_bit/scripts/analyze_learned_envelope_ablations.py",
        "```", "",
    ]
    (REPORTS / "LEARNED_ENVELOPE_ABLATIONS.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print(display.to_string(index=False))


if __name__ == "__main__":
    main()
