"""Fit residual-slack models and freeze the prospective cost-13 protocol."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor, export_text


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parents[1]
ARTIFACTS = EXPERIMENT / "artifacts"
REPORTS = EXPERIMENT / "reports"
SLACK = ROOT / "experiments" / "slack_bound_discovery"
PROTOCOL_PATH = EXPERIMENT / "protocol.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def predict_slack(model, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    return np.maximum(0.0, model.predict(frame[features]))


def calibrated_reduction(prediction: np.ndarray, true_slack: np.ndarray,
                         calibration: bool = True) -> tuple[np.ndarray, float]:
    offset = float(max(0.0, np.max(prediction - true_slack))) if calibration else 0.0
    return np.maximum(0.0, prediction - offset), offset


def model_specs(seed: int) -> dict[str, HistGradientBoostingRegressor]:
    common = dict(
        learning_rate=0.05,
        max_iter=350,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=3.0,
        early_stopping=False,
        random_state=seed,
    )
    return {
        "squared_boosting": HistGradientBoostingRegressor(
            loss="squared_error", **common),
        "lower_quantile_boosting": HistGradientBoostingRegressor(
            loss="quantile", quantile=0.10, **common),
    }


def main() -> None:
    base_protocol_bytes = PROTOCOL_PATH.read_bytes()
    protocol = json.loads(base_protocol_bytes)
    base_protocol_hash = hashlib.sha256(base_protocol_bytes).hexdigest()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    slack = load_module(
        "slack_discovery_core", SLACK / "scripts" / "run_slack_discovery.py")
    next_step = slack.load_module(
        "five_bit_next_step_for_residual",
        ROOT / "experiments" / "five_bit_next_step" / "scripts"
        / "run_aon_upper_tail.py")
    four = slack.prepare_four(next_step)
    five, cost12 = slack.prepare_five(next_step)
    four["residual_slack"] = (
        four.construction_upper_k_plus_1 - four.exact_k_plus_1)
    five["residual_slack"] = (
        five.construction_upper_k_plus_1 - five.exact_k_plus_1)
    cost12["residual_slack"] = (
        cost12.construction_upper_k_plus_1 - cost12.exact_k_plus_1)

    rng = np.random.default_rng(protocol["split_seed"])
    discovery_mask = rng.random(len(four)) < 0.80
    discovery = pd.concat([
        four[discovery_mask], five[five.exact_minimum_gates <= 11]
    ], ignore_index=True)
    selection = four[~discovery_mask].copy()
    features = slack.FEATURES

    selection_rows = []
    candidates = model_specs(protocol["model_seed"])
    for name, model in candidates.items():
        model.fit(discovery[features], discovery.residual_slack)
        prediction = predict_slack(model, selection, features)
        reduction, offset = calibrated_reduction(
            prediction, selection.residual_slack.to_numpy(float))
        selection_rows.append({
            "model": name,
            "selection_functions": len(selection),
            "selection_offset": offset,
            "selection_coverage": float(np.mean(
                reduction <= selection.residual_slack.to_numpy(float) + 1e-10)),
            "selection_mean_reduction": float(np.mean(reduction)),
            "selection_functions_improved": int(np.sum(reduction > 1e-12)),
        })
    selection_summary = pd.DataFrame(selection_rows)
    selected_name = selection_summary.sort_values(
        ["selection_mean_reduction", "model"], ascending=[False, True]
    ).iloc[0].model

    # Refit the selected teacher on all known pre-cost-12 exact targets.
    final_training = pd.concat([four, five[five.exact_minimum_gates <= 11]],
                               ignore_index=True)
    teacher = model_specs(protocol["model_seed"])[selected_name]
    teacher.fit(final_training[features], final_training.residual_slack)
    cost12_prediction = predict_slack(teacher, cost12, features)
    cost12_reduction, cost12_offset = calibrated_reduction(
        cost12_prediction, cost12.residual_slack.to_numpy(float))

    # Distill the fitted teacher, then calibrate the student independently.
    teacher_training_prediction = predict_slack(teacher, final_training, features)
    student = DecisionTreeRegressor(
        max_depth=5, min_samples_leaf=100,
        random_state=protocol["model_seed"] + 1)
    student.fit(final_training[features], teacher_training_prediction)
    student_cost12_prediction = predict_slack(student, cost12, features)
    student_reduction, student_offset = calibrated_reduction(
        student_cost12_prediction, cost12.residual_slack.to_numpy(float))

    cost12_results = []
    for model_name, prediction, reduction, offset in (
        ("teacher", cost12_prediction, cost12_reduction, cost12_offset),
        ("student", student_cost12_prediction, student_reduction, student_offset),
    ):
        current = cost12.copy()
        current["model"] = model_name
        current["predicted_residual_slack"] = prediction
        current["calibration_offset"] = offset
        current["learned_reduction_beyond_construction"] = reduction
        current["refined_upper_k_plus_1"] = (
            current.construction_upper_k_plus_1 - reduction)
        current["covered"] = (
            current.exact_k_plus_1 <= current.refined_upper_k_plus_1 + 1e-10)
        cost12_results.append(current)
    cost12_detail = pd.concat(cost12_results, ignore_index=True)
    cost12_summary = pd.DataFrame([{
        "model": model_name,
        "cohort": cohort,
        "functions": len(group),
        "calibration_offset": float(group.calibration_offset.iloc[0]),
        "coverage": float(group.covered.mean()),
        "mean_construction_upper": float(group.construction_upper_k_plus_1.mean()),
        "mean_refined_upper": float(group.refined_upper_k_plus_1.mean()),
        "mean_reduction_beyond_construction": float(
            group.learned_reduction_beyond_construction.mean()),
        "functions_improved": int(np.sum(
            group.learned_reduction_beyond_construction > 1e-12)),
    } for (model_name, cohort), group in cost12_detail.groupby(
        ["model", "cohort"], sort=False)])

    # Exhaustive finite-domain audit and raw-error counterexample mining.
    audits = []
    counterexamples = []
    known_five = pd.concat([five, cost12], ignore_index=True)
    for model_name, model in (("teacher", teacher), ("student", student)):
        four_prediction = predict_slack(model, four, features)
        four_reduction, audit_offset = calibrated_reduction(
            four_prediction, four.residual_slack.to_numpy(float))
        audited_upper = four.construction_upper_k_plus_1.to_numpy(float) - four_reduction
        audits.append({
            "model": model_name,
            "functions": len(four),
            "audit_offset": audit_offset,
            "coverage": float(np.mean(four.exact_k_plus_1 <= audited_upper + 1e-10)),
            "mean_reduction_beyond_construction": float(np.mean(four_reduction)),
            "functions_improved": int(np.sum(four_reduction > 1e-12)),
            "maximum_reduction": float(np.max(four_reduction)),
        })
        five_prediction = predict_slack(model, known_five, features)
        dangerous = five_prediction - known_five.residual_slack.to_numpy(float)
        top = np.argsort(dangerous)[-25:][::-1]
        for index in top:
            counterexamples.append({
                "model": model_name,
                "truth_table": int(known_five.iloc[index].truth_table),
                "exact_minimum_gates": int(
                    known_five.iloc[index].exact_minimum_gates),
                "cohort": known_five.iloc[index].cohort,
                "true_residual_slack": float(
                    known_five.iloc[index].residual_slack),
                "predicted_residual_slack": float(five_prediction[index]),
                "dangerous_residual": float(dangerous[index]),
            })
    audit = pd.DataFrame(audits)
    counterexample_frame = pd.DataFrame(counterexamples)

    teacher_path = ARTIFACTS / "residual_slack_teacher.joblib"
    student_path = ARTIFACTS / "residual_slack_student.joblib"
    joblib.dump(teacher, teacher_path)
    joblib.dump(student, student_path)
    (ARTIFACTS / "student_rules.txt").write_text(
        export_text(student, feature_names=features, decimals=3), encoding="utf-8")
    selection_summary.to_csv(ARTIFACTS / "model_selection.csv", index=False)
    cost12_detail.to_csv(ARTIFACTS / "cost12_calibration.csv", index=False)
    cost12_summary.to_csv(ARTIFACTS / "cost12_calibration_summary.csv", index=False)
    audit.to_csv(ARTIFACTS / "four_bit_residual_audit.csv", index=False)
    counterexample_frame.to_csv(ARTIFACTS / "known_counterexamples.csv", index=False)

    frozen = {
        "base_protocol_sha256": base_protocol_hash,
        "selected_model": selected_name,
        "features": features,
        "teacher_cost12_offset": cost12_offset,
        "student_cost12_offset": student_offset,
        "teacher_artifact": teacher_path.name,
        "teacher_sha256": sha256(teacher_path),
        "student_artifact": student_path.name,
        "student_sha256": sha256(student_path),
        "runtime": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "cost13": {
            "exact_cost": 13,
            "candidate_pool_size": 1000,
            "targeted_size": 80,
            "reference_size": 40,
            "selection_seed": protocol["cost13_selection_seed"],
            "selection": "largest teacher calibrated residual-slack reduction plus seeded random reference from remainder",
            "success_criteria": protocol["success_criteria"],
        },
    }
    frozen_path = ARTIFACTS / "frozen_cost13_protocol.json"
    frozen_path.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    frozen_hash = sha256(frozen_path)
    analysis = {
        "base_protocol_sha256": base_protocol_hash,
        "frozen_cost13_protocol_sha256": frozen_hash,
        "selected_model": selected_name,
        "training_functions": len(final_training),
        "selection_summary": selection_summary.to_dict(orient="records"),
        "cost12_summary": cost12_summary.to_dict(orient="records"),
        "four_bit_audit": audit.to_dict(orient="records"),
    }
    (ARTIFACTS / "residual_refinement_analysis.json").write_text(
        json.dumps(analysis, indent=2), encoding="utf-8")

    selection_display = selection_summary.round(4)
    cost12_display = cost12_summary.round(4)
    audit_display = audit.round(4)
    lines = [
        "# Residual-slack refinement below certified constructions", "",
        "This model predicts only the slack remaining below the certified",
        "restriction/factor upper bound. Costs through 11 are development data;",
        "cost 12 sets the one-sided safety correction; cost 13 remains untouched.", "",
        f"Base protocol SHA-256: `{base_protocol_hash}`", "",
        f"Frozen cost-13 protocol SHA-256: `{frozen_hash}`", "",
        "## Model selection", "", selection_display.to_markdown(index=False), "",
        f"Selected teacher: `{selected_name}`.", "",
        "## Cost-12 safety calibration", "", cost12_display.to_markdown(index=False), "",
        "Coverage is 100% by construction because the maximum dangerous cost-12",
        "residual is the frozen correction. The meaningful quantities are how much",
        "positive reduction survives and whether it appears in both cohorts.", "",
        "## Exhaustive four-input counterexample audit", "",
        audit_display.to_markdown(index=False), "",
        "The maximum dangerous residual over the complete universe gives each row",
        "a 100% finite-domain certificate. Known five-input raw counterexamples are",
        "saved separately and were included before the cost-13 freeze.", "",
        "## Frozen prospective test", "",
        "The serialized teacher, feature order, cost-12 correction, selection seed,",
        "cohort sizes, and success criteria are recorded in",
        "`../artifacts/frozen_cost13_protocol.json`. Any cost-13 evaluation must verify",
        "the recorded artifact hashes before enumeration.", "",
        "## Reproduce", "", "```powershell",
        "python experiments/residual_slack_refinement/scripts/run_local_refinement.py",
        "```", "",
    ]
    (REPORTS / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(selection_display.to_string(index=False))
    print(cost12_display.to_string(index=False))
    print(audit_display.to_string(index=False))


if __name__ == "__main__":
    main()
