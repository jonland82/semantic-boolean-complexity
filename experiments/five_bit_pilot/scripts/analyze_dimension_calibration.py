"""Evaluate frozen five-bit calibration on the prospective exact cost-10 layer."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
PILOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PILOT / "artifacts"
REPORTS = PILOT / "reports"


def load_core():
    path = ROOT / "experiments" / "four_bit" / "scripts" / "analyze_learned_envelope.py"
    spec = importlib.util.spec_from_file_location("learned_envelope_core", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metrics(group: pd.DataFrame, lower: np.ndarray, upper: np.ndarray) -> dict[str, float]:
    exact = group.exact_minimum_gates.to_numpy(float) + 1
    classical_lower = group.classical_lower_k_plus_1.to_numpy(float)
    classical_upper = group.classical_upper_k_plus_1.to_numpy(float)
    classical_width = classical_upper - classical_lower
    learned_width = upper - lower
    return {
        "coverage": float(np.mean((lower <= exact) & (exact <= upper))),
        "mean_learned_width": float(np.mean(learned_width)),
        "mean_relative_shrinkage": float(np.mean(1 - learned_width / classical_width)),
        "misses_below": int(np.sum(exact < lower)),
        "misses_above": int(np.sum(exact > upper)),
    }


def main() -> None:
    core = load_core()
    table = pd.read_csv(ARTIFACTS / "five_bit_pilot_predictions.csv")
    rows = []
    detail_frames = []
    probability = 0.975

    for language, language_table in table.groupby("language", sort=False):
        calibration = language_table[language_table.exact_minimum_gates <= 8]
        test = language_table[language_table.exact_minimum_gates == 10].copy()
        calibration_exact = calibration.exact_minimum_gates.to_numpy(float) + 1
        calibration_prediction = calibration.predicted_k_plus_1.to_numpy(float)
        score_minus = np.maximum(0.0, calibration_prediction - calibration_exact)
        score_plus = np.maximum(0.0, calibration_exact - calibration_prediction)
        allowance_minus = core.conformal_higher_quantile(score_minus, probability)
        allowance_plus = core.conformal_higher_quantile(score_plus, probability)

        prediction = test.predicted_k_plus_1.to_numpy(float)
        classical_lower = test.classical_lower_k_plus_1.to_numpy(float)
        classical_upper = test.classical_upper_k_plus_1.to_numpy(float)
        recalibrated_lower = np.maximum(classical_lower, prediction - allowance_minus)
        recalibrated_upper = np.minimum(classical_upper, prediction + allowance_plus)
        zero_shot_lower = test.calibrated_lower_k_plus_1.to_numpy(float)
        zero_shot_upper = test.calibrated_upper_k_plus_1.to_numpy(float)

        zero_shot = metrics(test, zero_shot_lower, zero_shot_upper)
        recalibrated = metrics(test, recalibrated_lower, recalibrated_upper)
        for method, values in (("n4_zero_shot", zero_shot),
                               ("n5_cost_0_8_calibrated", recalibrated)):
            rows.append({
                "language": language,
                "method": method,
                "calibration_functions": 0 if method == "n4_zero_shot" else len(calibration),
                "test_functions": len(test),
                "lower_allowance": (float(test.n4_lower_allowance.iloc[0])
                                    if method == "n4_zero_shot" else allowance_minus),
                "upper_allowance": (float(test.n4_upper_allowance.iloc[0])
                                    if method == "n4_zero_shot" else allowance_plus),
                **values,
            })

        # A minimal target-dimension adaptation: fit an affine correction of
        # K-hat on costs 0--7, reserve cost 8 solely for residual calibration,
        # and retain cost 10 as the prospective evaluation layer. Cost 9 is
        # unused by the frozen affine method.
        adaptation = language_table[language_table.exact_minimum_gates <= 7]
        affine_calibration = language_table[
            language_table.exact_minimum_gates == 8]
        design = np.column_stack((
            np.ones(len(adaptation)),
            adaptation.predicted_k_plus_1.to_numpy(float),
        ))
        coefficients, *_ = np.linalg.lstsq(
            design,
            adaptation.exact_minimum_gates.to_numpy(float) + 1,
            rcond=None,
        )
        calibration_center = (
            coefficients[0]
            + coefficients[1]
            * affine_calibration.predicted_k_plus_1.to_numpy(float))
        affine_exact = affine_calibration.exact_minimum_gates.to_numpy(float) + 1
        affine_minus = core.conformal_higher_quantile(
            np.maximum(0.0, calibration_center - affine_exact), probability)
        affine_plus = core.conformal_higher_quantile(
            np.maximum(0.0, affine_exact - calibration_center), probability)
        test_center = coefficients[0] + coefficients[1] * prediction
        affine_lower = np.maximum(classical_lower, test_center - affine_minus)
        affine_upper = np.minimum(classical_upper, test_center + affine_plus)
        affine_values = metrics(test, affine_lower, affine_upper)
        rows.append({
            "language": language,
            "method": "n5_affine_khat_adaptation",
            "calibration_functions": len(affine_calibration),
            "test_functions": len(test),
            "lower_allowance": affine_minus,
            "upper_allowance": affine_plus,
            **affine_values,
        })
        test["recalibrated_lower_k_plus_1"] = recalibrated_lower
        test["recalibrated_upper_k_plus_1"] = recalibrated_upper
        test["recalibrated_covered"] = (
            (recalibrated_lower <= test.exact_minimum_gates.to_numpy(float) + 1)
            & (test.exact_minimum_gates.to_numpy(float) + 1 <= recalibrated_upper))
        test["affine_predicted_k_plus_1"] = test_center
        test["affine_lower_k_plus_1"] = affine_lower
        test["affine_upper_k_plus_1"] = affine_upper
        test["affine_covered"] = (
            (affine_lower <= test.exact_minimum_gates.to_numpy(float) + 1)
            & (test.exact_minimum_gates.to_numpy(float) + 1 <= affine_upper))
        detail_frames.append(test)

    summary = pd.DataFrame(rows)
    details = pd.concat(detail_frames, ignore_index=True)
    summary.to_csv(ARTIFACTS / "five_bit_dimension_calibration_summary.csv", index=False)
    details.to_csv(ARTIFACTS / "five_bit_cost_10_holdout.csv", index=False)

    display = summary.copy()
    for column in ("lower_allowance", "upper_allowance", "coverage",
                   "mean_learned_width", "mean_relative_shrinkage"):
        display[column] = display[column].round(4)
    lines = [
        "# Five-input dimension calibration", "",
        "The four-input model and point predictions remain frozen. Exact five-input",
        "samples from cost layers 0--8 calibrate only the two residual allowances;",
        "the previously unseen exact cost-10 layer is held out for evaluation.", "",
        display.to_markdown(index=False), "",
        "This is a small, cost-layer-balanced feasibility test. It demonstrates",
        "whether a modest amount of target-dimension calibration repairs zero-shot",
        "coverage. The affine variant fits a correction of K-hat on costs 0--7,",
        "uses cost 8 only for calibration, ignores cost 9, and tests on cost 10.",
        "This",
        "cost-layer-balanced pilot is not representative of arbitrary functions.",
        "", "## Reproduce", "", "```powershell",
        "python experiments/five_bit_pilot/scripts/analyze_dimension_calibration.py",
        "```", "",
    ]
    (REPORTS / "DIMENSION_CALIBRATION.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print(display.to_string(index=False))


if __name__ == "__main__":
    main()
