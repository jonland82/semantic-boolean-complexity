"""Learn calibrated corrections to the existing AON analytic upper bound."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, export_text


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parents[1]
ARTIFACTS = EXPERIMENT / "artifacts"
REPORTS = EXPERIMENT / "reports"
FOUR = ROOT / "experiments" / "four_bit" / "artifacts"
PILOT = ROOT / "experiments" / "five_bit_pilot" / "artifacts"
NEXT = ROOT / "experiments" / "five_bit_next_step"
PROTOCOL_PATH = EXPERIMENT / "protocol.json"

FEATURES = [
    "inputs",
    "khrapchenko_product",
    "decision_tree_leaf_count",
    "mean_certificate_size",
    "maximum_certificate_size",
    "mean_certificate_size_output_0",
    "mean_certificate_size_output_1",
    *[f"anf_terms_degree_{degree}" for degree in range(5)],
    *[f"fourier_negative_degree_{degree}" for degree in range(5)],
    "classical_width",
    "and_factorable_partitions",
    "or_factorable_partitions",
    "canalizing_variable_count",
    "restriction_gain",
    "factor_gain",
]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def add_bound_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["classical_width"] = (
        result.existing_upper_k_plus_1 - result.classical_lower_k_plus_1)
    result["restriction_gain"] = np.maximum(
        0.0, result.existing_upper_k_plus_1
        - result.restriction_upper_k_plus_1)
    result["factor_gain"] = np.maximum(
        0.0, result.existing_upper_k_plus_1
        - result.factor_upper_k_plus_1)
    result["construction_upper_k_plus_1"] = np.minimum.reduce([
        result.existing_upper_k_plus_1.to_numpy(float),
        result.restriction_upper_k_plus_1.to_numpy(float),
        result.factor_upper_k_plus_1.to_numpy(float),
    ])
    result["exact_k_plus_1"] = result.exact_minimum_gates + 1
    result["true_slack"] = (
        result.existing_upper_k_plus_1 - result.exact_k_plus_1)
    return result


def prepare_four(next_step) -> pd.DataFrame:
    features = pd.read_csv(FOUR / "feature_table.csv")
    positions = pd.read_csv(FOUR / "complexity_sandwich_positions.csv")
    costs = features.target_AND_OR_NOT.to_numpy(int)
    structure = [next_step.construction_features(int(table), 4, costs)
                 for table in features.truth_table]
    structure = pd.DataFrame(structure)
    result = features[[
        "truth_table", "khrapchenko_product", "decision_tree_leaf_count",
        "mean_certificate_size", "maximum_certificate_size",
        "mean_certificate_size_output_0", "mean_certificate_size_output_1",
        *[f"anf_terms_degree_{degree}" for degree in range(5)],
        *[f"fourier_negative_degree_{degree}" for degree in range(5)],
    ]].copy()
    result["inputs"] = 4
    result["exact_minimum_gates"] = features.target_AND_OR_NOT
    result["classical_lower_k_plus_1"] = positions.lower_AND_OR_NOT
    result["existing_upper_k_plus_1"] = positions.upper_AND_OR_NOT
    for column in ("and_factorable_partitions", "or_factorable_partitions",
                   "canalizing_variable_count", "restriction_upper_k_plus_1",
                   "factor_upper_k_plus_1"):
        result[column] = structure[column]
    result["source"] = "n4_exhaustive"
    result["cohort"] = "n4"
    return add_bound_columns(result)


def prepare_five(next_step) -> tuple[pd.DataFrame, pd.DataFrame]:
    pilot = pd.read_csv(PILOT / "five_bit_pilot_predictions.csv")
    pilot = pilot[pilot.language == "AND_OR_NOT"].copy()
    pilot = next_step.add_structure(
        pilot, pd.read_csv(FOUR / "feature_table.csv").target_AND_OR_NOT.to_numpy(int))
    pilot["inputs"] = 5
    pilot["existing_upper_k_plus_1"] = pilot.classical_upper_k_plus_1
    pilot["source"] = "n5_sparse_layers"
    pilot["cohort"] = "cost_" + pilot.exact_minimum_gates.astype(str)
    pilot = add_bound_columns(pilot)

    cost12 = pd.read_csv(NEXT / "artifacts" / "aon_cost_12_selected.csv")
    cost12["inputs"] = 5
    cost12["existing_upper_k_plus_1"] = cost12.classical_upper_k_plus_1
    cost12["source"] = "n5_cost_12_challenge"
    cost12["cohort"] = "cost_12_" + cost12.cohort
    cost12 = add_bound_columns(cost12)
    return pilot, cost12


def models(seed: int):
    return {
        "boosting": HistGradientBoostingRegressor(
            loss="squared_error", learning_rate=0.06, max_iter=300,
            max_leaf_nodes=15, min_samples_leaf=30,
            l2_regularization=2.0, early_stopping=False,
            random_state=seed),
        "shallow_tree": DecisionTreeRegressor(
            max_depth=5, min_samples_leaf=100, random_state=seed + 1),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
    }


def evaluate(model_name: str, model, calibration: pd.DataFrame,
             tests: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    cal_prediction = np.maximum(0.0, model.predict(calibration[FEATURES]))
    dangerous_residual = cal_prediction - calibration.true_slack.to_numpy(float)
    offset = float(np.max(dangerous_residual))
    result = tests.copy()
    result["model"] = model_name
    result["predicted_slack"] = np.maximum(0.0, model.predict(result[FEATURES]))
    result["calibration_offset"] = offset
    result["learned_reduction"] = np.maximum(
        0.0, result.predicted_slack - offset)
    result["learned_upper_k_plus_1"] = (
        result.existing_upper_k_plus_1 - result.learned_reduction)
    result["combined_upper_k_plus_1"] = np.minimum(
        result.construction_upper_k_plus_1,
        result.learned_upper_k_plus_1)
    result["covered"] = (
        result.exact_k_plus_1 <= result.learned_upper_k_plus_1 + 1e-10)
    return result, offset


def summarize(group: pd.DataFrame) -> dict[str, object]:
    return {
        "model": group.model.iloc[0],
        "cohort": group.cohort.iloc[0],
        "functions": len(group),
        "coverage": float(group.covered.mean()),
        "misses": int((~group.covered).sum()),
        "mean_existing_upper": float(group.existing_upper_k_plus_1.mean()),
        "mean_construction_upper": float(
            group.construction_upper_k_plus_1.mean()),
        "mean_learned_upper": float(group.learned_upper_k_plus_1.mean()),
        "mean_combined_upper": float(group.combined_upper_k_plus_1.mean()),
        "mean_reduction": float(group.learned_reduction.mean()),
        "functions_improved": int(np.sum(group.learned_reduction > 1e-12)),
        "functions_beyond_construction": int(np.sum(
            group.learned_upper_k_plus_1
            < group.construction_upper_k_plus_1 - 1e-12)),
        "mean_reduction_beyond_construction": float(np.mean(
            group.construction_upper_k_plus_1 - group.combined_upper_k_plus_1)),
        "maximum_violation": float(np.maximum(
            0.0, group.exact_k_plus_1 - group.learned_upper_k_plus_1).max()),
    }


def main() -> None:
    protocol_bytes = PROTOCOL_PATH.read_bytes()
    protocol = json.loads(protocol_bytes)
    protocol_hash = hashlib.sha256(protocol_bytes).hexdigest()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    next_step = load_module(
        "five_bit_next_step_core", NEXT / "scripts" / "run_aon_upper_tail.py")
    four = prepare_four(next_step)
    five, cost12 = prepare_five(next_step)
    rng = np.random.default_rng(protocol["split_seed"])
    discovery_mask = rng.random(len(four)) < 0.80
    discovery = pd.concat([
        four[discovery_mask], five[five.exact_minimum_gates <= 9]
    ], ignore_index=True)
    calibration = five[five.exact_minimum_gates == 10].copy()
    cost11 = five[five.exact_minimum_gates == 11].copy()
    tests = pd.concat([cost11, cost12], ignore_index=True)

    detail_frames = []
    offsets = {}
    fitted = models(protocol["model_seed"])
    for name, model in fitted.items():
        model.fit(discovery[FEATURES], discovery.true_slack)
        evaluated, offset = evaluate(name, model, calibration, tests)
        detail_frames.append(evaluated)
        offsets[name] = offset
    details = pd.concat(detail_frames, ignore_index=True)
    summary = pd.DataFrame([
        summarize(group) for _, group in details.groupby(["model", "cohort"], sort=False)
    ])

    audits = []
    for name, model in fitted.items():
        predicted = np.maximum(0.0, model.predict(four[FEATURES]))
        audit_offset = float(np.max(predicted - four.true_slack.to_numpy(float)))
        reduction = np.maximum(0.0, predicted - audit_offset)
        audited_upper = four.existing_upper_k_plus_1.to_numpy(float) - reduction
        construction_upper = four.construction_upper_k_plus_1.to_numpy(float)
        combined_upper = np.minimum(audited_upper, construction_upper)
        audits.append({
            "model": name,
            "functions": len(four),
            "audit_offset": audit_offset,
            "coverage": float(np.mean(four.exact_k_plus_1 <= audited_upper + 1e-10)),
            "mean_reduction": float(np.mean(reduction)),
            "functions_improved": int(np.sum(reduction > 1e-12)),
            "maximum_reduction": float(np.max(reduction)),
            "combined_coverage": float(np.mean(
                four.exact_k_plus_1 <= combined_upper + 1e-10)),
            "mean_combined_reduction": float(np.mean(
                four.existing_upper_k_plus_1 - combined_upper)),
            "functions_beyond_construction": int(np.sum(
                audited_upper < construction_upper - 1e-12)),
            "mean_reduction_beyond_construction": float(np.mean(
                construction_upper - combined_upper)),
        })
    audit = pd.DataFrame(audits)

    tree_rules = export_text(
        fitted["shallow_tree"], feature_names=FEATURES, decimals=3)
    (ARTIFACTS / "shallow_tree_rules.txt").write_text(tree_rules, encoding="utf-8")
    details.to_csv(ARTIFACTS / "five_bit_slack_predictions.csv", index=False)
    summary.to_csv(ARTIFACTS / "five_bit_slack_summary.csv", index=False)
    audit.to_csv(ARTIFACTS / "four_bit_finite_audit.csv", index=False)
    metadata = {
        "protocol_sha256": protocol_hash,
        "protocol": protocol,
        "features": FEATURES,
        "training_functions": len(discovery),
        "calibration_functions": len(calibration),
        "calibration_offsets": offsets,
        "five_bit_summary": summary.to_dict(orient="records"),
        "four_bit_audit": audit.to_dict(orient="records"),
    }
    (ARTIFACTS / "slack_discovery_analysis.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")

    display = summary.copy()
    for column in ("coverage", "mean_existing_upper", "mean_construction_upper",
                   "mean_learned_upper", "mean_combined_upper", "mean_reduction",
                   "mean_reduction_beyond_construction", "maximum_violation"):
        display[column] = display[column].round(4)
    audit_display = audit.copy()
    for column in ("audit_offset", "coverage", "mean_reduction",
                   "maximum_reduction", "combined_coverage",
                   "mean_combined_reduction", "mean_reduction_beyond_construction"):
        audit_display[column] = audit_display[column].round(4)
    lines = [
        "# ML-assisted AON upper-bound slack discovery", "",
        "The model predicts removable slack in the existing analytic upper bound.",
        "A maximum one-sided residual from five-input cost 10 is subtracted from",
        "the prediction before any reduction is made. Costs 11 and 12 are evaluation",
        "only; the paper and previous experimental artifacts are unchanged.", "",
        f"Protocol SHA-256: `{protocol_hash}`", "",
        f"Training functions: {len(discovery):,}; calibration functions: {len(calibration)}.", "",
        "## Five-input validation", "", display.to_markdown(index=False), "",
        "Coverage here is prospective or held-out empirical coverage, not a universal",
        "guarantee. A miss means the model claimed more removable slack than existed.", "",
        "The combined endpoint is the minimum of the learned upper bound and the",
        "certified restriction/factor construction. `functions_beyond_construction`",
        "counts cases where ML supplies a further reduction; its coverage remains",
        "the learned bound's empirical coverage.", "",
        "Boosting is the strongest robust candidate here: it covers both cost-12",
        "cohorts completely and covers 39/40 cost-11 functions. Combined with the",
        "certified construction, it improves a further 15/40 cost-11 functions and",
        "13/40 cost-12 random-reference functions, but none of the selected cost-12",
        "upper-tail functions. The interpretable tree is slightly more aggressive",
        "and correspondingly has more misses.", "",
        "## Exhaustive four-input audit", "", audit_display.to_markdown(index=False), "",
        "The audit offset is the worst dangerous residual across all 65,536 four-input",
        "functions. These audited bounds therefore have a complete finite-domain",
        "certificate even though the underlying slack predictions are learned.", "",
        "Because the exhaustively audited learned bound and the construction bound",
        "are both valid on four inputs, their combined minimum is also a certified",
        "finite-domain bound. `mean_reduction_beyond_construction` isolates the part",
        "of that improvement attributable only to the learned correction.", "",
        "For boosting, the combined bound remains valid on all 65,536 functions,",
        "reduces the old upper endpoint by 1.3048 gates on average, and beats the",
        "construction-only endpoint on 22,408 functions. The ML-only increment over",
        "construction averages 0.1829 gates across the complete universe.", "",
        "## Interpretable candidate", "",
        "The complete depth-5 regression-tree rule list is stored in",
        "`../artifacts/shallow_tree_rules.txt`. It is a discovery artifact rather",
        "than a proved dimension-general inequality.", "",
        "## Reproduce", "", "```powershell",
        "python experiments/slack_bound_discovery/scripts/run_slack_discovery.py",
        "```", "",
    ]
    (REPORTS / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(display.to_string(index=False))
    print(audit_display.to_string(index=False))


if __name__ == "__main__":
    main()
