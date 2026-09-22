"""Frozen conditional uncertainty model for prospective five-bit cost 11."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


PILOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PILOT / "artifacts"
REPORTS = PILOT / "reports"
TRAIN_MAX_COST = 9
CALIBRATION_COST = 10
TEST_COST = 11
QUANTILE = 0.90
SEED = 9173
MODEL_SPEC = {
    "loss": "quantile",
    "quantile": QUANTILE,
    "learning_rate": 0.06,
    "max_iter": 250,
    "max_leaf_nodes": 10,
    "min_samples_leaf": 15,
    "l2_regularization": 2.0,
    "early_stopping": False,
}
FEATURES = [
    "predicted_k_plus_1",
    "khrapchenko_product",
    "decision_tree_leaf_count",
    "mean_certificate_size",
    "maximum_certificate_size",
    "mean_certificate_size_output_0",
    "mean_certificate_size_output_1",
    *[f"anf_terms_degree_{degree}" for degree in range(5)],
    *[f"fourier_negative_degree_{degree}" for degree in range(5)],
    "classical_width",
]


def main() -> None:
    table = pd.read_csv(ARTIFACTS / "five_bit_pilot_predictions.csv")
    table["classical_width"] = (
        table.classical_upper_k_plus_1 - table.classical_lower_k_plus_1)
    if int(table.exact_minimum_gates.max()) < TEST_COST:
        raise RuntimeError(
            f"pilot reaches cost {int(table.exact_minimum_gates.max())}; "
            f"cost {TEST_COST} is required")

    rows = []
    details = []
    for language_index, (language, group) in enumerate(
            table.groupby("language", sort=False)):
        train = group[group.exact_minimum_gates <= TRAIN_MAX_COST]
        calibration = group[group.exact_minimum_gates == CALIBRATION_COST]
        test = group[group.exact_minimum_gates == TEST_COST].copy()
        exact_train = train.exact_minimum_gates.to_numpy(float) + 1
        exact_calibration = calibration.exact_minimum_gates.to_numpy(float) + 1
        exact_test = test.exact_minimum_gates.to_numpy(float) + 1
        allowances = []
        for side_index, side in enumerate(("lower", "upper")):
            if side == "lower":
                train_score = np.maximum(
                    0.0, train.predicted_k_plus_1.to_numpy(float) - exact_train)
                calibration_score = np.maximum(
                    0.0, calibration.predicted_k_plus_1.to_numpy(float)
                    - exact_calibration)
            else:
                train_score = np.maximum(
                    0.0, exact_train - train.predicted_k_plus_1.to_numpy(float))
                calibration_score = np.maximum(
                    0.0, exact_calibration
                    - calibration.predicted_k_plus_1.to_numpy(float))
            model = HistGradientBoostingRegressor(
                **MODEL_SPEC,
                random_state=SEED + 100 * language_index + side_index,
            )
            model.fit(train[FEATURES], train_score)
            calibration_residual = (
                calibration_score - model.predict(calibration[FEATURES]))
            # With 40 calibration functions, the finite-sample 97.5% tail
            # quantile is the maximum residual. This is fixed before cost 11.
            conformal_offset = float(np.max(calibration_residual))
            allowance = np.maximum(
                0.0, model.predict(test[FEATURES]) + conformal_offset)
            allowances.append(allowance)

        prediction = test.predicted_k_plus_1.to_numpy(float)
        lower = np.maximum(
            test.classical_lower_k_plus_1.to_numpy(float),
            prediction - allowances[0])
        upper = np.minimum(
            test.classical_upper_k_plus_1.to_numpy(float),
            prediction + allowances[1])
        covered = (lower <= exact_test) & (exact_test <= upper)
        learned_width = upper - lower
        classical_width = test.classical_width.to_numpy(float)
        rows.append({
            "language": language,
            "train_functions": len(train),
            "calibration_functions": len(calibration),
            "prospective_test_functions": len(test),
            "coverage": float(np.mean(covered)),
            "mean_lower_allowance": float(np.mean(allowances[0])),
            "mean_upper_allowance": float(np.mean(allowances[1])),
            "mean_learned_width": float(np.mean(learned_width)),
            "mean_classical_width": float(np.mean(classical_width)),
            "mean_relative_shrinkage": float(
                np.mean(1 - learned_width / classical_width)),
            "misses_below": int(np.sum(exact_test < lower)),
            "misses_above": int(np.sum(exact_test > upper)),
        })
        test["conditional_lower_k_plus_1"] = lower
        test["conditional_upper_k_plus_1"] = upper
        test["conditional_covered"] = covered
        test["conditional_lower_allowance"] = allowances[0]
        test["conditional_upper_allowance"] = allowances[1]
        details.append(test)

    summary = pd.DataFrame(rows)
    detail = pd.concat(details, ignore_index=True)
    summary.to_csv(ARTIFACTS / "five_bit_conditional_calibration_summary.csv", index=False)
    detail.to_csv(ARTIFACTS / "five_bit_cost_11_prospective.csv", index=False)
    protocol = {
        "train_max_cost": TRAIN_MAX_COST,
        "calibration_cost": CALIBRATION_COST,
        "test_cost": TEST_COST,
        "features": FEATURES,
        "model": MODEL_SPEC,
        "seed": SEED,
        "tails": "separate one-sided scores; maximum cost-10 residual correction",
    }
    protocol_hash = hashlib.sha256(
        json.dumps(protocol, sort_keys=True).encode("utf-8")).hexdigest()
    (ARTIFACTS / "five_bit_conditional_calibration.json").write_text(
        json.dumps({"protocol": protocol, "protocol_sha256": protocol_hash,
                    "summary": summary.to_dict(orient="records")}, indent=2),
        encoding="utf-8")

    display = summary.copy()
    for column in display.columns[4:]:
        display[column] = display[column].astype(float).round(4)
    lines = [
        "# Prospective conditional calibration at five inputs", "",
        "The protocol was frozen before exact cost-11 enumeration. It trains",
        "separate conditional lower/upper error models through cost 9, uses cost",
        "10 only for conformal residual correction, and evaluates cost 11 once.",
        "K-hat remains the interval center; all uncertainty features are semantic",
        "and observable without knowing exact K.", "", display.to_markdown(index=False),
        "", f"Protocol SHA-256: `{protocol_hash}`", "",
        "## Reproduce", "", "```powershell",
        "python experiments/five_bit_pilot/scripts/analyze_conditional_calibration.py",
        "```", "",
    ]
    (REPORTS / "CONDITIONAL_CALIBRATION.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print(display.to_string(index=False))


if __name__ == "__main__":
    main()
