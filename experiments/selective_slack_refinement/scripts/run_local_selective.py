"""Fit selective residual-slack refinement and freeze cost-14 evaluation."""

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
PROTOCOL_PATH = EXPERIMENT / "protocol.json"
RESIDUAL = ROOT / "experiments" / "residual_slack_refinement"
SLACK = ROOT / "experiments" / "slack_bound_discovery"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_teacher(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=0.05, max_iter=400,
        max_leaf_nodes=15, min_samples_leaf=30, l2_regularization=3.0,
        early_stopping=False, random_state=seed)


def stratum(prediction: np.ndarray) -> np.ndarray:
    return np.where(prediction <= 2.0, "low",
                    np.where(prediction <= 3.0, "medium", "high"))


def fit_offsets(prediction: np.ndarray, true_slack: np.ndarray,
                buffer: float) -> dict[str, float]:
    groups = stratum(prediction)
    offsets = {}
    for name in ("low", "medium", "high"):
        selected = groups == name
        if not np.any(selected):
            raise RuntimeError(f"calibration stratum {name} is empty")
        dangerous = prediction[selected] - true_slack[selected]
        offsets[name] = float(max(0.0, np.max(dangerous)) + buffer)
    return offsets


def apply_policy(prediction: np.ndarray, offsets: dict[str, float],
                 minimum_reduction: float) -> tuple[np.ndarray, np.ndarray]:
    groups = stratum(prediction)
    correction = np.asarray([offsets[name] for name in groups])
    reduction = np.maximum(0.0, prediction - correction)
    reduction[reduction < minimum_reduction] = 0.0
    return reduction, groups


def add_residual(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["residual_slack"] = (
        result.construction_upper_k_plus_1 - result.exact_k_plus_1)
    return result


def main() -> None:
    protocol_bytes = PROTOCOL_PATH.read_bytes()
    protocol = json.loads(protocol_bytes)
    protocol_hash = hashlib.sha256(protocol_bytes).hexdigest()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    slack = load_module(
        "slack_for_selective", SLACK / "scripts" / "run_slack_discovery.py")
    next_step = slack.load_module(
        "next_for_selective",
        ROOT / "experiments" / "five_bit_next_step" / "scripts"
        / "run_aon_upper_tail.py")
    four = add_residual(slack.prepare_four(next_step))
    five, cost12 = slack.prepare_five(next_step)
    five = add_residual(five)
    cost12 = add_residual(cost12)
    cost13 = add_residual(pd.read_csv(
        RESIDUAL / "artifacts" / "cost13_prospective_predictions.csv"))
    features = slack.FEATURES

    training = pd.concat([four, five, cost12], ignore_index=True)
    teacher = make_teacher(protocol["model_seed"])
    teacher.fit(training[features], training.residual_slack)

    prior = json.loads((
        RESIDUAL / "artifacts" / "cost13_prospective_analysis.json").read_text())
    transition_buffer = float(prior["overall"]["maximum_violation"])
    calibration_prediction = np.maximum(
        0.0, teacher.predict(cost13[features]))
    teacher_offsets = fit_offsets(
        calibration_prediction, cost13.residual_slack.to_numpy(float),
        transition_buffer)
    teacher_reduction, teacher_strata = apply_policy(
        calibration_prediction, teacher_offsets,
        protocol["minimum_applied_reduction"])

    training_prediction = np.maximum(0.0, teacher.predict(training[features]))
    student = DecisionTreeRegressor(
        max_depth=5, min_samples_leaf=100,
        random_state=protocol["model_seed"] + 1)
    student.fit(training[features], training_prediction)
    student_calibration_prediction = np.maximum(
        0.0, student.predict(cost13[features]))
    student_offsets = fit_offsets(
        student_calibration_prediction, cost13.residual_slack.to_numpy(float),
        transition_buffer)
    student_reduction, student_strata = apply_policy(
        student_calibration_prediction, student_offsets,
        protocol["minimum_applied_reduction"])

    detail_frames = []
    summary_rows = []
    for model_name, prediction, reduction, groups, offsets in (
        ("teacher", calibration_prediction, teacher_reduction,
         teacher_strata, teacher_offsets),
        ("student", student_calibration_prediction, student_reduction,
         student_strata, student_offsets),
    ):
        current = cost13.copy()
        current["model"] = model_name
        current["risk_stratum"] = groups
        current["predicted_residual_slack_v3"] = prediction
        current["selective_reduction"] = reduction
        current["selective_upper_k_plus_1"] = (
            current.construction_upper_k_plus_1 - reduction)
        current["covered_v3"] = (
            current.exact_k_plus_1 <= current.selective_upper_k_plus_1 + 1e-10)
        detail_frames.append(current)
        for (cohort, risk), group in current.groupby(
                ["cohort", "risk_stratum"], sort=False):
            summary_rows.append({
                "model": model_name,
                "cohort": cohort,
                "risk_stratum": risk,
                "functions": len(group),
                "offset": offsets[risk],
                "coverage": float(group.covered_v3.mean()),
                "mean_reduction": float(group.selective_reduction.mean()),
                "functions_improved": int(np.sum(group.selective_reduction > 0)),
            })
    calibration_detail = pd.concat(detail_frames, ignore_index=True)
    calibration_summary = pd.DataFrame(summary_rows)

    audits = []
    for model_name, model in (("teacher", teacher), ("student", student)):
        prediction = np.maximum(0.0, model.predict(four[features]))
        groups = stratum(prediction)
        audit_offsets = {}
        for risk in ("low", "medium", "high"):
            selected = groups == risk
            audit_offsets[risk] = float(max(
                0.0, np.max(prediction[selected]
                            - four.residual_slack.to_numpy(float)[selected])))
        reduction, _ = apply_policy(
            prediction, audit_offsets, protocol["minimum_applied_reduction"])
        upper = four.construction_upper_k_plus_1.to_numpy(float) - reduction
        audits.append({
            "model": model_name,
            "functions": len(four),
            "coverage": float(np.mean(four.exact_k_plus_1 <= upper + 1e-10)),
            "mean_reduction": float(np.mean(reduction)),
            "functions_improved": int(np.sum(reduction > 0)),
            "low_offset": audit_offsets["low"],
            "medium_offset": audit_offsets["medium"],
            "high_offset": audit_offsets["high"],
        })
    audit = pd.DataFrame(audits)

    teacher_path = ARTIFACTS / "selective_teacher.joblib"
    student_path = ARTIFACTS / "selective_student.joblib"
    joblib.dump(teacher, teacher_path)
    joblib.dump(student, student_path)
    (ARTIFACTS / "student_rules.txt").write_text(
        export_text(student, feature_names=features, decimals=3), encoding="utf-8")
    calibration_detail.to_csv(ARTIFACTS / "cost13_selective_calibration.csv", index=False)
    calibration_summary.to_csv(
        ARTIFACTS / "cost13_selective_calibration_summary.csv", index=False)
    audit.to_csv(ARTIFACTS / "four_bit_selective_audit.csv", index=False)

    frozen = {
        "base_protocol_sha256": protocol_hash,
        "features": features,
        "teacher_artifact": teacher_path.name,
        "teacher_sha256": sha256(teacher_path),
        "student_artifact": student_path.name,
        "student_sha256": sha256(student_path),
        "teacher_offsets": teacher_offsets,
        "student_offsets": student_offsets,
        "transition_buffer": transition_buffer,
        "minimum_applied_reduction": protocol["minimum_applied_reduction"],
        "strata": {"low_max": 2.0, "medium_max": 3.0},
        "runtime": {
            "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "cost14": {
            "exact_cost": 14,
            **protocol["cost14"],
            "selection_seed": protocol["cost14_selection_seed"],
            "success_criteria": protocol["success_criteria"],
        },
    }
    frozen_path = ARTIFACTS / "frozen_cost14_protocol.json"
    frozen_path.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    frozen_hash = sha256(frozen_path)
    analysis = {
        "base_protocol_sha256": protocol_hash,
        "frozen_cost14_protocol_sha256": frozen_hash,
        "training_functions": len(training),
        "transition_buffer": transition_buffer,
        "teacher_offsets": teacher_offsets,
        "calibration_summary": calibration_summary.to_dict(orient="records"),
        "four_bit_audit": audit.to_dict(orient="records"),
    }
    (ARTIFACTS / "selective_refinement_analysis.json").write_text(
        json.dumps(analysis, indent=2), encoding="utf-8")

    lines = [
        "# Selective residual-slack refinement", "",
        "The teacher is trained through cost 12. Exact cost-13 results set separate",
        "low/medium/high safety corrections, each enlarged by the previously observed",
        "cross-layer maximum-residual drift. Reductions below 0.25 are suppressed.", "",
        f"Base protocol SHA-256: `{protocol_hash}`", "",
        f"Frozen cost-14 protocol SHA-256: `{frozen_hash}`", "",
        f"Cross-layer buffer: {transition_buffer:.6f}.", "",
        "## Cost-13 calibration", "",
        calibration_summary.round(4).to_markdown(index=False), "",
        "## Exhaustive four-input audit", "",
        audit.round(4).to_markdown(index=False), "",
        "## Frozen next test", "",
        "The model hashes, feature order, stratum thresholds, offsets, abstention",
        "threshold, sampling seed, cohort sizes, and success criteria are stored in",
        "`../artifacts/frozen_cost14_protocol.json`.", "",
    ]
    (REPORTS / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(calibration_summary.round(4).to_string(index=False))
    print(audit.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
