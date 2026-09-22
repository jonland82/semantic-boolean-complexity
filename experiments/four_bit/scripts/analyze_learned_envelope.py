"""Build prediction-centered complexity envelopes with honest class holdout.

The point estimate is the compact sandwich model from the envelopes paper. It
predicts normalized position between the classical lower and upper endpoints.
This analysis asks whether a residual allowance around that estimate shrinks
the classical interval while retaining exact minimum gate count.
"""

from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold


EXPERIMENT = Path(__file__).resolve().parents[1]
ARTIFACTS = EXPERIMENT / "artifacts"
FIGURES = EXPERIMENT / "figures"
REPORTS = EXPERIMENT / "reports"

LANGUAGES = ["NAND", "NOR", "AND_OR_NOT"]
LABELS = {"NAND": "NAND", "NOR": "NOR", "AND_OR_NOT": "AND/OR/NOT"}
FOLDS = 5
SEED = 0
COVERAGES = (0.90, 0.95, 0.99)
PROTOCOL_VERSION = "learned-envelope-v2-gap-normalized"
MODEL_SPEC = {
    "learning_rate": 0.06,
    "max_iter": 250,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 10,
    "l2_regularization": 1.0,
    "early_stopping": False,
}

CERTIFICATES = [
    "mean_certificate_size",
    "maximum_certificate_size",
    "mean_certificate_size_output_0",
    "mean_certificate_size_output_1",
]
ALGEBRAIC_PHASE = (
    [f"anf_terms_degree_{degree}" for degree in range(5)]
    + [f"fourier_negative_degree_{degree}" for degree in range(5)]
)
COMPACT = ["khrapchenko_product", "decision_tree_leaf_count"] + CERTIFICATES + ALGEBRAIC_PHASE


def make_model(language_index: int, fold: int, purpose: int) -> HistGradientBoostingRegressor:
    """Return the frozen compact-sandwich regressor used throughout the audit."""
    return HistGradientBoostingRegressor(
        **MODEL_SPEC,
        random_state=SEED + 100 * language_index + 10 * purpose + fold,
    )


def conformal_higher_quantile(scores: np.ndarray, probability: float) -> float:
    """Finite-sample split-conformal quantile with the conservative higher rule."""
    if len(scores) == 0:
        raise ValueError("calibration scores must not be empty")
    rank = min(len(scores), math.ceil((len(scores) + 1) * probability))
    return float(np.partition(scores, rank - 1)[rank - 1])


def class_layout(table: pd.DataFrame):
    values = table[COMPACT].to_numpy(float)
    unique, representative, groups = np.unique(
        values, axis=0, return_index=True, return_inverse=True)
    folds = list(KFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(unique))
    fold_of_class = np.empty(len(unique), dtype=np.int8)
    for fold, (_, test) in enumerate(folds):
        fold_of_class[test] = fold
    return unique, representative, groups, folds, fold_of_class


def class_means(values: np.ndarray, groups: np.ndarray, classes: int):
    weights = np.bincount(groups, minlength=classes).astype(float)
    means = np.bincount(groups, weights=values, minlength=classes) / weights
    return weights, means


def class_worst_scores(prediction: np.ndarray, exact: np.ndarray,
                       groups: np.ndarray, class_indices: np.ndarray,
                       scale: np.ndarray | None = None):
    """Return worst lower/upper miss for each requested semantic class."""
    lower_error = np.maximum(0.0, prediction - exact)
    upper_error = np.maximum(0.0, exact - prediction)
    if scale is not None:
        positive = scale > 1e-12
        lower_error = np.divide(
            lower_error, scale, out=np.zeros_like(lower_error), where=positive)
        upper_error = np.divide(
            upper_error, scale, out=np.zeros_like(upper_error), where=positive)
    lower_scores = np.zeros(len(class_indices), dtype=float)
    upper_scores = np.zeros(len(class_indices), dtype=float)
    for output_index, class_index in enumerate(class_indices):
        members = groups == class_index
        lower_scores[output_index] = float(np.max(lower_error[members]))
        upper_scores[output_index] = float(np.max(upper_error[members]))
    return lower_scores, upper_scores


def interval_metrics(exact: np.ndarray, classical_lower: np.ndarray,
                     classical_upper: np.ndarray, learned_lower: np.ndarray,
                     learned_upper: np.ndarray, groups: np.ndarray | None = None):
    tolerance = 1e-10
    covered = ((learned_lower <= exact + tolerance)
               & (learned_upper >= exact - tolerance))
    classical_width = classical_upper - classical_lower
    learned_width = np.maximum(0.0, learned_upper - learned_lower)
    positive = classical_width > tolerance
    relative = np.zeros(len(exact), dtype=float)
    relative[positive] = 1 - learned_width[positive] / classical_width[positive]
    relative[~positive] = 1.0
    classical_integer_lower = np.ceil(classical_lower - tolerance)
    classical_integer_upper = np.floor(classical_upper + tolerance)
    learned_integer_lower = np.ceil(learned_lower - tolerance)
    learned_integer_upper = np.floor(learned_upper + tolerance)
    classical_integer_count = np.maximum(
        0.0, classical_integer_upper - classical_integer_lower + 1)
    learned_integer_count = np.maximum(
        0.0, learned_integer_upper - learned_integer_lower + 1)
    integer_reduction = np.zeros(len(exact), dtype=float)
    valid_integer = classical_integer_count > 0
    integer_reduction[valid_integer] = (
        1 - learned_integer_count[valid_integer]
        / classical_integer_count[valid_integer])
    result = {
        "function_coverage": float(np.mean(covered)),
        "mean_classical_width": float(np.mean(classical_width)),
        "mean_learned_width": float(np.mean(learned_width)),
        "mean_relative_shrinkage": float(np.mean(relative)),
        "median_relative_shrinkage": float(np.median(relative)),
        "fraction_strictly_narrower": float(
            np.mean(learned_width < classical_width - tolerance)),
        "mean_classical_integer_candidates": float(
            np.mean(classical_integer_count)),
        "mean_learned_integer_candidates": float(
            np.mean(learned_integer_count)),
        "mean_integer_candidate_reduction": float(np.mean(integer_reduction)),
    }
    if groups is not None:
        class_covered = np.ones(int(groups.max()) + 1, dtype=bool)
        for class_index in range(len(class_covered)):
            class_covered[class_index] = bool(np.all(covered[groups == class_index]))
        result["class_coverage"] = float(np.mean(class_covered))
    return result


def analyze_language(table: pd.DataFrame, language: str, language_index: int,
                     unique: np.ndarray, groups: np.ndarray, folds,
                     fold_of_class: np.ndarray):
    classes = len(unique)
    exact = table[f"target_{language}"].to_numpy(float) + 1
    lower = table[f"lower_{language}"].to_numpy(float)
    upper = table[f"upper_{language}"].to_numpy(float)
    width = upper - lower
    position = table[f"position_{language}"].to_numpy(float)
    weights, position_mean = class_means(position, groups, classes)

    # Each profile class is predicted by a model that saw the other four folds.
    predicted_position_by_class = np.empty(classes, dtype=float)
    for fold, (train, test) in enumerate(folds):
        model = make_model(language_index, fold, purpose=0)
        model.fit(unique[train], position_mean[train], sample_weight=weights[train])
        predicted_position_by_class[test] = np.clip(
            model.predict(unique[test]), 0, 1)
    predicted_position = predicted_position_by_class[groups]
    prediction = lower + predicted_position * width

    lower_error = np.maximum(0.0, prediction - exact)
    upper_error = np.maximum(0.0, exact - prediction)
    exhaustive_minus = float(np.max(lower_error))
    exhaustive_plus = float(np.max(upper_error))
    exhaustive_lower = np.maximum(lower, prediction - exhaustive_minus)
    exhaustive_upper = np.minimum(upper, prediction + exhaustive_plus)
    exhaustive_metrics = interval_metrics(
        exact, lower, upper, exhaustive_lower, exhaustive_upper, groups)

    rows = []
    calibrated_columns: dict[str, np.ndarray] = {}
    # For outer fold f, fold f+1 calibrates and the remaining three train.
    # Calibration scores are worst-case within each complete semantic class.
    for coverage in COVERAGES:
        learned_lower = np.empty(len(table), dtype=float)
        learned_upper = np.empty(len(table), dtype=float)
        normalized_lower = np.empty(len(table), dtype=float)
        normalized_upper = np.empty(len(table), dtype=float)
        allowances_minus = []
        allowances_plus = []
        normalized_minus = []
        normalized_plus = []
        alpha = 1 - coverage
        side_probability = 1 - alpha / 2
        for test_fold in range(FOLDS):
            calibration_fold = (test_fold + 1) % FOLDS
            test_classes = np.flatnonzero(fold_of_class == test_fold)
            calibration_classes = np.flatnonzero(fold_of_class == calibration_fold)
            train_classes = np.flatnonzero(
                (fold_of_class != test_fold) & (fold_of_class != calibration_fold))

            model = make_model(language_index, test_fold, purpose=1)
            model.fit(unique[train_classes], position_mean[train_classes],
                      sample_weight=weights[train_classes])
            class_position = np.clip(model.predict(unique), 0, 1)
            nested_prediction = lower + class_position[groups] * width
            calibration_minus, calibration_plus = class_worst_scores(
                nested_prediction, exact, groups, calibration_classes)
            calibration_normalized_minus, calibration_normalized_plus = (
                class_worst_scores(
                    nested_prediction, exact, groups, calibration_classes,
                    scale=width))
            allowance_minus = conformal_higher_quantile(
                calibration_minus, side_probability)
            allowance_plus = conformal_higher_quantile(
                calibration_plus, side_probability)
            q_minus = conformal_higher_quantile(
                calibration_normalized_minus, side_probability)
            q_plus = conformal_higher_quantile(
                calibration_normalized_plus, side_probability)
            allowances_minus.append(allowance_minus)
            allowances_plus.append(allowance_plus)
            normalized_minus.append(q_minus)
            normalized_plus.append(q_plus)

            test_members = np.isin(groups, test_classes)
            learned_lower[test_members] = np.maximum(
                lower[test_members], nested_prediction[test_members] - allowance_minus)
            learned_upper[test_members] = np.minimum(
                upper[test_members], nested_prediction[test_members] + allowance_plus)
            normalized_lower[test_members] = np.maximum(
                lower[test_members],
                nested_prediction[test_members] - q_minus * width[test_members])
            normalized_upper[test_members] = np.minimum(
                upper[test_members],
                nested_prediction[test_members] + q_plus * width[test_members])

        metrics = interval_metrics(
            exact, lower, upper, learned_lower, learned_upper, groups)
        metrics.update({
            "language": language,
            "method": "split_conformal_class_worst",
            "allowance_units": "target_units",
            "nominal_coverage": coverage,
            "mean_lower_allowance": float(np.mean(allowances_minus)),
            "mean_upper_allowance": float(np.mean(allowances_plus)),
            "maximum_lower_allowance": float(np.max(allowances_minus)),
            "maximum_upper_allowance": float(np.max(allowances_plus)),
        })
        rows.append(metrics)
        suffix = str(int(round(100 * coverage)))
        calibrated_columns[f"calibrated_{suffix}_lower_k_plus_1"] = learned_lower
        calibrated_columns[f"calibrated_{suffix}_upper_k_plus_1"] = learned_upper
        calibrated_columns[f"calibrated_{suffix}_integer_lower_k_plus_1"] = np.ceil(
            learned_lower - 1e-10)
        calibrated_columns[f"calibrated_{suffix}_integer_upper_k_plus_1"] = np.floor(
            learned_upper + 1e-10)

        normalized_metrics = interval_metrics(
            exact, lower, upper, normalized_lower, normalized_upper, groups)
        normalized_metrics.update({
            "language": language,
            "method": "split_conformal_gap_normalized",
            "allowance_units": "fraction_of_analytic_width",
            "nominal_coverage": coverage,
            "mean_lower_allowance": float(np.mean(normalized_minus)),
            "mean_upper_allowance": float(np.mean(normalized_plus)),
            "maximum_lower_allowance": float(np.max(normalized_minus)),
            "maximum_upper_allowance": float(np.max(normalized_plus)),
            "minimum_shrinkage_guarantee": float(max(
                0.0, 1 - max(
                    minus + plus
                    for minus, plus in zip(normalized_minus, normalized_plus)))),
        })
        rows.append(normalized_metrics)
        if coverage == 0.95:
            calibrated_columns["gap_normalized_95_lower_k_plus_1"] = normalized_lower
            calibrated_columns["gap_normalized_95_upper_k_plus_1"] = normalized_upper

    prediction_table = pd.DataFrame({
        "truth_table": table["truth_table"].to_numpy(int),
        "language": language,
        "compact_class": groups,
        "fold": fold_of_class[groups],
        "exact_k_plus_1": exact,
        "classical_lower_k_plus_1": lower,
        "classical_upper_k_plus_1": upper,
        "predicted_position": predicted_position,
        "predicted_k_plus_1": prediction,
        "signed_residual_exact_minus_prediction": exact - prediction,
        "exhaustive_lower_k_plus_1": exhaustive_lower,
        "exhaustive_upper_k_plus_1": exhaustive_upper,
        "exhaustive_integer_lower_k_plus_1": np.ceil(exhaustive_lower - 1e-10),
        "exhaustive_integer_upper_k_plus_1": np.floor(exhaustive_upper + 1e-10),
        **calibrated_columns,
    })
    exhaustive_row = {
        "language": language,
        "method": "cross_fitted_exhaustive_max_residual",
        "allowance_units": "target_units",
        "nominal_coverage": 1.0,
        "mean_lower_allowance": exhaustive_minus,
        "mean_upper_allowance": exhaustive_plus,
        "maximum_lower_allowance": exhaustive_minus,
        "maximum_upper_allowance": exhaustive_plus,
        **exhaustive_metrics,
    }
    return prediction_table, pd.DataFrame([exhaustive_row, *rows])


def make_figure(predictions: pd.DataFrame) -> None:
    rows = []
    for language in LANGUAGES:
        subset = predictions[predictions.language == language]
        classical = (
            np.floor(subset.classical_upper_k_plus_1.to_numpy(float) + 1e-10)
            - np.ceil(subset.classical_lower_k_plus_1.to_numpy(float) - 1e-10)
            + 1
        )
        learned = (
            np.floor(subset.calibrated_95_upper_k_plus_1.to_numpy(float) + 1e-10)
            - np.ceil(subset.calibrated_95_lower_k_plus_1.to_numpy(float) - 1e-10)
            + 1
        )
        rows.append((LABELS[language], float(np.mean(classical)),
                     float(np.mean(learned))))

    fig, axis = plt.subplots(figsize=(7.2, 3.35))
    y_positions = np.arange(len(rows))[::-1]
    for y, (label, classical, learned) in zip(y_positions, rows):
        axis.plot([learned, classical], [y, y], color="#9AA2A8", linewidth=3,
                  solid_capstyle="round", zorder=1)
        axis.scatter(classical, y, s=95, color="#535B61", zorder=2,
                     label="Theory-only envelope" if y == y_positions[0] else None)
        axis.scatter(learned, y, s=95, color="#E8872D", zorder=3,
                     label=r"Calibrated envelope around $\widehat K$"
                     if y == y_positions[0] else None)
        axis.text(classical + 0.35, y, f"{classical:.2f}", va="center",
                  color="#30363A", fontsize=9)
        axis.text(learned - 0.35, y, f"{learned:.2f}", va="center", ha="right",
                  color="#A94E08", fontsize=9, fontweight="bold")
        reduction = 100 * (1 - learned / classical)
        axis.text((learned + classical) / 2, y + 0.14,
                  f"{reduction:.0f}% fewer", ha="center", va="bottom",
                  fontsize=8, color="#535B61")

    axis.set_yticks(y_positions, [row[0] for row in rows])
    axis.set_xlabel("Mean number of gate counts still possible")
    axis.set_xlim(0, max(row[1] for row in rows) + 3)
    axis.set_ylim(-0.35, len(rows) - 0.55)
    axis.grid(axis="x", alpha=0.18)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    axis.legend(loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    for extension in ("png", "pdf", "svg"):
        fig.savefig(FIGURES / f"learned_envelope_shrinkage.{extension}",
                    dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(summary: pd.DataFrame, predictions: pd.DataFrame) -> None:
    display = summary.copy()
    numeric = [column for column in display.columns
               if column not in ("language", "method", "allowance_units")]
    display[numeric] = display[numeric].astype(float).round(4)
    largest = (
        predictions.assign(
            absolute_residual=lambda frame:
            frame.signed_residual_exact_minus_prediction.abs())
        .sort_values(["language", "absolute_residual"], ascending=[True, False])
        .groupby("language", sort=False).head(5)
        [["language", "truth_table", "exact_k_plus_1", "predicted_k_plus_1",
          "signed_residual_exact_minus_prediction", "classical_lower_k_plus_1",
          "classical_upper_k_plus_1"]]
        .copy()
    )
    largest["truth_table"] = largest.truth_table.map(lambda value: f"0x{value:04x}")
    for column in largest.columns[2:]:
        largest[column] = largest[column].astype(float).round(3)

    lines = [
        "# Learned semantic complexity envelope", "",
        "The compact sandwich model predicts normalized position between the",
        "Khrapchenko lower endpoint and the decision-tree/prime-cover upper endpoint.",
        "Every point prediction below is out-of-fold over complete 16-coordinate",
        "semantic-profile classes.", "", "## Interval results", "",
        display.to_markdown(index=False), "",
        "The exhaustive row uses the largest observed one-sided residual of all",
        "65,536 cross-fitted predictions. It is a finite audit of this fixed table,",
        "not a dimension-independent theorem. The calibrated rows use a stricter",
        "three-fold train, one-fold calibrate, one-fold test rotation. Calibration",
        "scores take the worst residual inside each complete semantic class; the two",
        "tails use a Bonferroni split of the stated joint error rate.", "",
        "For any single split, if the held-out profile class is exchangeable with",
        "the calibration classes, the finite-sample higher quantile gives at least",
        "the nominal simultaneous class-uniform coverage by the union bound. The",
        "five rotating splits reported here are an empirical aggregation of that",
        "grouped split-conformal construction.", "",
        "The gap-normalized rows divide each residual by the width of its analytic",
        "envelope before calibration. Their allowances are therefore fractions,",
        "not gate-count units. If the two fold-specific fractions sum to less than",
        "one, the analytic-gap theorem guarantees that every noncollapsed interval",
        "in that fold shrinks by at least one minus their sum, independently of",
        "whether its target is covered.", "",
        "Intervals are always intersected with the classical semantic envelope.",
        "Coverage therefore cannot be worse than an un-intersected learned interval.",
        "", "![Learned envelope shrinkage](../figures/learned_envelope_shrinkage.png)",
        "", "## Largest cross-fitted point errors", "",
        largest.to_markdown(index=False), "", "## Interpretation", "",
        "A useful learned envelope must jointly achieve high held-out coverage and",
        "material width reduction. Mean and median relative shrinkage are computed",
        "per function against the original interval; already-collapsed classical",
        "intervals receive shrinkage one by convention. Integer-candidate columns",
        "also round the endpoints to the feasible integral values of `K + 1`.", "",
        "The model supplies the interval center, while calibration or exhaustive",
        "verification supplies the error allowance. Only the classical endpoints",
        "remain dimension-independent mathematical bounds at this stage.", "",
        "## Reproduce", "", "```powershell",
        "python experiments/four_bit/scripts/analyze_learned_envelope.py",
        "```", "",
    ]
    (REPORTS / "LEARNED_ENVELOPE.md").write_text(
        "\n".join(lines), encoding="utf-8")


def main() -> None:
    features = pd.read_csv(ARTIFACTS / "feature_table.csv")
    positions = pd.read_csv(ARTIFACTS / "complexity_sandwich_positions.csv")
    envelope_columns = [
        "truth_table",
        *[f"lower_{language}" for language in LANGUAGES],
        *[f"upper_{language}" for language in LANGUAGES],
        *[f"position_{language}" for language in LANGUAGES],
    ]
    table = features.merge(
        positions[envelope_columns], on="truth_table", validate="one_to_one")
    unique, _, groups, folds, fold_of_class = class_layout(table)
    prediction_tables = []
    summaries = []
    for language_index, language in enumerate(LANGUAGES):
        prediction_table, summary = analyze_language(
            table, language, language_index, unique, groups, folds, fold_of_class)
        prediction_tables.append(prediction_table)
        summaries.append(summary)
    predictions = pd.concat(prediction_tables, ignore_index=True)
    summary = pd.concat(summaries, ignore_index=True)

    predictions.to_csv(ARTIFACTS / "learned_envelope_predictions.csv", index=False)
    summary.to_csv(ARTIFACTS / "learned_envelope_summary.csv", index=False)
    protocol = {
        "version": PROTOCOL_VERSION,
        "features": COMPACT,
        "model": MODEL_SPEC,
        "folds": FOLDS,
        "seed": SEED,
        "coverages": list(COVERAGES),
        "grouping": "exact compact semantic profile",
        "calibration_scores": [
            "worst one-sided residual within profile class",
            "worst one-sided residual divided by analytic width within profile class",
        ],
    }
    protocol_hash = hashlib.sha256(
        json.dumps(protocol, sort_keys=True).encode("utf-8")).hexdigest()
    analysis = {
        "functions": int(len(table)),
        "compact_profile_classes": int(len(unique)),
        "protocol": protocol,
        "protocol_sha256": protocol_hash,
        "summary": summary.to_dict(orient="records"),
    }
    (ARTIFACTS / "learned_envelope_analysis.json").write_text(
        json.dumps(analysis, indent=2), encoding="utf-8")
    make_figure(predictions)
    write_report(summary, predictions)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
