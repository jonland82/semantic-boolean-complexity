"""Retrospective local study of selection-aware, one-sided calibration.

Cost 12 trains a model for the dangerous residual, cost 13 calibrates its
one-sided error after model-based selection, and cost 14 is used only as an
already-observed diagnostic.  Nothing produced here is prospective evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parents[1]
ARTIFACTS = EXPERIMENT / "artifacts"
REPORTS = EXPERIMENT / "reports"
FROZEN_PATH = ARTIFACTS / "frozen_cost14_protocol.json"
FOUR_COSTS_PATH = ROOT / "experiments" / "four_bit" / "artifacts" / "feature_table.csv"
COST12_PATH = (
    ROOT / "experiments" / "residual_slack_refinement" / "artifacts"
    / "cost12_calibration.csv"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact_cofactor(table: int, variable: int, value: int) -> int:
    result = 0
    for reduced in range(16):
        low = reduced & ((1 << variable) - 1)
        assignment = low | ((reduced >> variable) << (variable + 1))
        assignment |= value << variable
        result |= ((table >> assignment) & 1) << reduced
    return result


def add_restriction_witnesses(frame: pd.DataFrame, four_costs: np.ndarray) -> pd.DataFrame:
    result = frame.copy()
    rows: list[dict[str, float]] = []
    for raw_table in result.truth_table:
        table = int(raw_table)
        pairs = []
        for variable in range(5):
            left = int(four_costs[compact_cofactor(table, variable, 0)])
            right = int(four_costs[compact_cofactor(table, variable, 1)])
            pairs.append((left, right))
        totals = np.asarray([left + right for left, right in pairs])
        best = int(totals.min())
        best_pairs = [pair for pair, total in zip(pairs, totals) if total == best]
        second = int(np.partition(totals, 1)[1])
        rows.append({
            "restriction_best_cofactor_min": float(min(min(pair) for pair in best_pairs)),
            "restriction_best_cofactor_max": float(max(max(pair) for pair in best_pairs)),
            "restriction_best_imbalance": float(max(abs(a - b) for a, b in best_pairs)),
            "restriction_best_witnesses": float(len(best_pairs)),
            "restriction_second_best_gap": float(second - best),
        })
    witness = pd.DataFrame(rows, index=result.index)
    for column in witness:
        result[column] = witness[column]
    return result


def normalize_cohort(frame: pd.DataFrame) -> pd.Series:
    return frame.cohort.astype(str).str.contains("target|upper_tail", case=False)


def prepare_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    frozen = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
    teacher_path = ARTIFACTS / frozen["teacher_artifact"]
    if sha256(teacher_path) != frozen["teacher_sha256"]:
        raise RuntimeError("frozen teacher hash mismatch")
    teacher = joblib.load(teacher_path)
    model_features = frozen["features"]

    cost12 = pd.read_csv(COST12_PATH)
    cost12 = cost12[cost12.model == "teacher"].copy()
    cost13 = pd.read_csv(ARTIFACTS / "cost13_selective_calibration.csv")
    cost13 = cost13[cost13.model == "teacher"].copy()
    cost14 = pd.read_csv(ARTIFACTS / "cost14_prospective_predictions.csv")

    cost12["score"] = np.maximum(0.0, teacher.predict(cost12[model_features]))
    cost13["score"] = cost13.predicted_residual_slack_v3
    cost14["score"] = cost14.predicted_residual_slack
    for frame in (cost12, cost13, cost14):
        frame["headroom"] = frame.construction_upper_k_plus_1 - frame.exact_k_plus_1
        frame["dangerous_residual"] = frame.score - frame.headroom
        frame["targeted"] = normalize_cohort(frame)

    four_costs = pd.read_csv(FOUR_COSTS_PATH).target_AND_OR_NOT.to_numpy(int)
    return (
        add_restriction_witnesses(cost12, four_costs),
        add_restriction_witnesses(cost13, four_costs),
        add_restriction_witnesses(cost14, four_costs),
        frozen,
    )


BASE_FEATURES = ["score"]
CONSTRUCTION_FEATURES = [
    "score", "construction_upper_k_plus_1", "restriction_upper_k_plus_1",
    "restriction_gain", "factor_gain", "decision_tree_leaf_count",
]
WITNESS_FEATURES = CONSTRUCTION_FEATURES + [
    "restriction_best_cofactor_min", "restriction_best_cofactor_max",
    "restriction_best_imbalance", "restriction_best_witnesses",
    "restriction_second_best_gap", "khrapchenko_product",
    "mean_certificate_size", "maximum_certificate_size",
    "anf_terms_degree_1", "anf_terms_degree_2", "anf_terms_degree_3",
    "fourier_negative_degree_2",
]


def fit_error_model(training: pd.DataFrame, features: list[str]) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        loss="quantile", quantile=0.9, learning_rate=0.05, max_iter=250,
        max_leaf_nodes=7, min_samples_leaf=12, l2_regularization=5.0,
        early_stopping=False, random_state=31418,
    )
    model.fit(training[features], training.dangerous_residual)
    return model


def evaluate_policy(
    name: str,
    model: HistGradientBoostingRegressor,
    features: list[str],
    calibration: pd.DataFrame,
    evaluation: pd.DataFrame,
    calibration_targeted_only: bool,
    minimum_reduction: float,
) -> tuple[dict, pd.DataFrame]:
    calibration_base = model.predict(calibration[features])
    selected = calibration.targeted.to_numpy(bool) if calibration_targeted_only else np.ones(len(calibration), bool)
    calibration_shortfall = calibration.dangerous_residual.to_numpy(float) - calibration_base
    additive_margin = float(max(0.0, calibration_shortfall[selected].max()))

    predicted_correction = np.maximum(0.0, model.predict(evaluation[features]) + additive_margin)
    reduction = np.maximum(0.0, evaluation.score.to_numpy(float) - predicted_correction)
    reduction[reduction < minimum_reduction] = 0.0
    violation = np.maximum(0.0, reduction - evaluation.headroom.to_numpy(float))
    detail = evaluation[["truth_table", "cohort", "targeted", "score", "headroom"]].copy()
    detail["policy"] = name
    detail["predicted_correction"] = predicted_correction
    detail["reduction"] = reduction
    detail["violation"] = violation
    detail["covered"] = violation <= 1e-10

    targeted = detail.targeted.to_numpy(bool)
    row = {
        "policy": name,
        "calibration_targeted_only": calibration_targeted_only,
        "additive_margin": additive_margin,
        "coverage": float(detail.covered.mean()),
        "targeted_coverage": float(detail.loc[targeted, "covered"].mean()),
        "random_coverage": float(detail.loc[~targeted, "covered"].mean()),
        "misses": int((~detail.covered).sum()),
        "maximum_violation": float(detail.violation.max()),
        "mean_reduction": float(detail.reduction.mean()),
        "targeted_mean_reduction": float(detail.loc[targeted, "reduction"].mean()),
        "improved_fraction": float((detail.reduction > 0).mean()),
    }
    return row, detail


def frozen_baseline(cost14: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    detail = cost14[["truth_table", "cohort", "targeted", "score", "headroom"]].copy()
    source = cost14.reset_index(drop=True)
    detail["policy"] = "frozen_baseline"
    detail["predicted_correction"] = source.score - source.selective_reduction
    detail["reduction"] = source.selective_reduction
    detail["violation"] = source.violation
    detail["covered"] = source.covered.astype(bool)
    targeted = detail.targeted.to_numpy(bool)
    return ({
        "policy": "frozen_baseline",
        "calibration_targeted_only": False,
        "additive_margin": np.nan,
        "coverage": float(detail.covered.mean()),
        "targeted_coverage": float(detail.loc[targeted, "covered"].mean()),
        "random_coverage": float(detail.loc[~targeted, "covered"].mean()),
        "misses": int((~detail.covered).sum()),
        "maximum_violation": float(detail.violation.max()),
        "mean_reduction": float(detail.reduction.mean()),
        "targeted_mean_reduction": float(detail.loc[targeted, "reduction"].mean()),
        "improved_fraction": float((detail.reduction > 0).mean()),
    }, detail)


def guarded_frozen_policy(
    name: str, cost14: pd.DataFrame, *, unit_cap: bool = False,
    abstain_endpoint: float | None = None,
) -> tuple[dict, pd.DataFrame]:
    detail = cost14[["truth_table", "cohort", "targeted", "score", "headroom"]].copy()
    reduction = cost14.selective_reduction.to_numpy(float).copy()
    if unit_cap:
        reduction = np.minimum(reduction, 1.0)
    if abstain_endpoint is not None:
        reduction[cost14.construction_upper_k_plus_1.to_numpy(float) <= abstain_endpoint] = 0.0
    violation = np.maximum(0.0, reduction - cost14.headroom.to_numpy(float))
    detail["policy"] = name
    detail["predicted_correction"] = detail.score.to_numpy(float) - reduction
    detail["reduction"] = reduction
    detail["violation"] = violation
    detail["covered"] = violation <= 1e-10
    targeted = detail.targeted.to_numpy(bool)
    return ({
        "policy": name,
        "calibration_targeted_only": False,
        "additive_margin": np.nan,
        "coverage": float(detail.covered.mean()),
        "targeted_coverage": float(detail.loc[targeted, "covered"].mean()),
        "random_coverage": float(detail.loc[~targeted, "covered"].mean()),
        "misses": int((~detail.covered).sum()),
        "maximum_violation": float(detail.violation.max()),
        "mean_reduction": float(detail.reduction.mean()),
        "targeted_mean_reduction": float(detail.loc[targeted, "reduction"].mean()),
        "improved_fraction": float((detail.reduction > 0).mean()),
    }, detail)


def main() -> None:
    cost12, cost13, cost14, frozen = prepare_data()
    minimum_reduction = float(frozen["minimum_applied_reduction"])
    rows: list[dict] = []
    details: list[pd.DataFrame] = []
    baseline_row, baseline_detail = frozen_baseline(cost14)
    rows.append(baseline_row)
    details.append(baseline_detail)
    for guard_name, guard_args in (
        ("frozen_unit_reduction_cap", {"unit_cap": True}),
        ("frozen_abstain_endpoint_16", {"abstain_endpoint": 16.0}),
    ):
        guard_row, guard_detail = guarded_frozen_policy(
            guard_name, cost14, **guard_args
        )
        rows.append(guard_row)
        details.append(guard_detail)

    specifications = [
        ("score_only", BASE_FEATURES),
        ("construction_aware", CONSTRUCTION_FEATURES),
        ("full_restriction_witness", WITNESS_FEATURES),
    ]
    transport_rows = []
    for specification, features in specifications:
        model12 = fit_error_model(cost12, features)
        for targeted_only in (False, True):
            suffix = "selected" if targeted_only else "all"
            name = f"{specification}_{suffix}"
            row, detail = evaluate_policy(
                name, model12, features, cost13, cost14,
                targeted_only, minimum_reduction,
            )
            rows.append(row)
            details.append(detail)

            # Historical transport check: fit the same rule and calibrate its
            # maximum shortfall inside cost 12, then test cost 13.
            base12 = model12.predict(cost12[features])
            pool = cost12.targeted.to_numpy(bool) if targeted_only else np.ones(len(cost12), bool)
            shortfall12 = cost12.dangerous_residual.to_numpy(float) - base12
            margin12 = float(max(0.0, shortfall12[pool].max()))
            correction13 = np.maximum(0.0, model12.predict(cost13[features]) + margin12)
            reduction13 = np.maximum(0.0, cost13.score.to_numpy(float) - correction13)
            reduction13[reduction13 < minimum_reduction] = 0.0
            violation13 = np.maximum(0.0, reduction13 - cost13.headroom.to_numpy(float))
            transport_rows.append({
                "policy": name,
                "cost12_margin": margin12,
                "cost13_coverage": float(np.mean(violation13 <= 1e-10)),
                "cost13_misses": int(np.sum(violation13 > 1e-10)),
                "cost13_mean_reduction": float(np.mean(reduction13)),
                "cost13_maximum_violation": float(np.max(violation13)),
            })

    summary = pd.DataFrame(rows)
    baseline_mean = float(summary.loc[
        summary.policy == "frozen_baseline", "mean_reduction"
    ].iloc[0])
    summary["fraction_of_baseline_gain"] = summary.mean_reduction / baseline_mean
    detail = pd.concat(details, ignore_index=True)
    transport = pd.DataFrame(transport_rows)
    summary.to_csv(ARTIFACTS / "local_selection_aware_summary.csv", index=False)
    detail.to_csv(ARTIFACTS / "local_selection_aware_predictions.csv", index=False)
    transport.to_csv(ARTIFACTS / "local_selection_aware_transport.csv", index=False)

    best = summary[summary.policy != "frozen_baseline"].sort_values(
        ["misses", "mean_reduction"], ascending=[True, False]
    ).iloc[0]
    selected_calibration_n = int(cost13.targeted.sum())
    future_selected_n = int(cost14.targeted.sum())
    exchangeable_batch_coverage = selected_calibration_n / (
        selected_calibration_n + future_selected_n
    )
    calibration_needed_for_99 = math.ceil(99 * future_selected_n)
    best_json = json.loads(pd.DataFrame([best]).to_json(orient="records"))[0]
    policies_json = json.loads(summary.to_json(orient="records"))
    transport_json = json.loads(transport.to_json(orient="records"))
    analysis = {
        "status": "retrospective diagnostic; not prospective evidence",
        "training_layer": 12,
        "calibration_layer": 13,
        "diagnostic_layer": 14,
        "cost12_functions": len(cost12),
        "cost13_functions": len(cost13),
        "cost14_functions": len(cost14),
        "candidate_pool_replay_available": False,
        "max_residual_batch_calibration": {
            "selected_calibration_cases": selected_calibration_n,
            "future_selected_cases": future_selected_n,
            "exchangeable_probability_future_batch_max_is_below_calibration_max": exchangeable_batch_coverage,
            "selected_calibration_cases_needed_for_99_percent_batch_probability": calibration_needed_for_99,
        },
        "best_diagnostic_policy": best_json,
        "policies": policies_json,
        "historical_transport": transport_json,
    }
    (ARTIFACTS / "local_selection_aware_analysis.json").write_text(
        json.dumps(analysis, indent=2), encoding="utf-8"
    )

    display = summary[[
        "policy", "coverage", "misses", "maximum_violation",
        "mean_reduction", "fraction_of_baseline_gain", "improved_fraction",
    ]].copy()
    lines = [
        "# Local selection-aware calibration study", "",
        "This is a retrospective diagnostic. Cost 14 has already been observed,",
        "so these results cannot validate a repaired rule prospectively.", "",
        "## Design", "",
        "A 90th-quantile model predicts the dangerous residual `score - true headroom`",
        "from cost 12. Cost 13 supplies a maximum one-sided additive correction.",
        "The repaired reduction is applied unchanged to cost 14. We compare score-only,",
        "construction-aware, and full restriction-witness feature sets; each correction",
        "is calibrated either on all cost-13 cases or only the model-selected cohort.",
        "The original 1,000-candidate pools were not retained, so full top-80 replay is",
        "not available locally.", "",
        "A maximum-residual correction based on 80 selected calibration cases has",
        "only `80 / (80 + 80) = 50%` probability of exceeding the maximum of a",
        "future 80-case batch under ideal exchangeability. Reaching 99% by this",
        "rank argument requires at least 7,920 selected calibration cases.", "",
        "## Cost-14 diagnostic", "",
        display.round(4).to_markdown(index=False), "",
        "## Historical cost-12 to cost-13 transport", "",
        transport.round(4).to_markdown(index=False), "",
        "## Interpretation", "",
        f"The strongest diagnostic policy is `{best.policy}`: it has",
        f"{int(best.misses)} misses, {best.mean_reduction:.4f} mean reduction, and",
        f"improves {100 * best.improved_fraction:.1f}% of the 120 cases.",
        "The learned dangerous-residual variants transport worse than the frozen",
        "baseline. The only zero-miss local variants are conservative boundary",
        "guards: capping any attempted refinement at one gate, or abstaining at",
        "construction endpoint 16. The unit cap retains most of the original gain",
        "and is the more general hypothesis; endpoint-16 abstention is tied to this",
        "observed layer.",
        "Because the six failures motivated this study, the result is hypothesis",
        "generation only. A repaired policy must be chosen without reference to a",
        "future labeled cohort, frozen, and then tested on new exact data.", "",
    ]
    (REPORTS / "LOCAL_SELECTION_AWARE_RESULTS.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(display.round(4).to_string(index=False))
    print("\nHistorical transport")
    print(transport.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
