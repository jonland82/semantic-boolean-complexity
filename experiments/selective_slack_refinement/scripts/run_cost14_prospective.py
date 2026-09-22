"""Run the frozen prospective selective residual-slack test at exact cost 14."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parents[1]
ARTIFACTS = EXPERIMENT / "artifacts"
REPORTS = EXPERIMENT / "reports"
FOUR = ROOT / "experiments" / "four_bit" / "artifacts"
FROZEN_PATH = ARTIFACTS / "frozen_cost14_protocol.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply_policy(prediction: np.ndarray, frozen: dict) -> tuple[np.ndarray, np.ndarray]:
    low_max = frozen["strata"]["low_max"]
    medium_max = frozen["strata"]["medium_max"]
    groups = np.where(prediction <= low_max, "low",
                      np.where(prediction <= medium_max, "medium", "high"))
    correction = np.asarray([frozen["teacher_offsets"][name] for name in groups])
    reduction = np.maximum(0.0, prediction - correction)
    reduction[reduction < frozen["minimum_applied_reduction"]] = 0.0
    return reduction, groups


def main() -> None:
    frozen_hash = sha256(FROZEN_PATH)
    frozen = json.loads(FROZEN_PATH.read_text())
    observed_runtime = {
        "numpy": np.__version__, "pandas": pd.__version__,
        "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }
    if observed_runtime != frozen["runtime"]:
        raise RuntimeError(
            f"runtime mismatch: expected {frozen['runtime']}, observed {observed_runtime}")
    teacher_path = ARTIFACTS / frozen["teacher_artifact"]
    if sha256(teacher_path) != frozen["teacher_sha256"]:
        raise RuntimeError("frozen teacher artifact hash mismatch")
    teacher = joblib.load(teacher_path)

    pilot = load_module(
        "pilot_for_cost14",
        ROOT / "experiments" / "five_bit_pilot" / "scripts" / "run_local_pilot.py")
    next_step = load_module(
        "next_for_cost14",
        ROOT / "experiments" / "five_bit_next_step" / "scripts"
        / "run_aon_upper_tail.py")
    slack = load_module(
        "slack_for_cost14",
        ROOT / "experiments" / "slack_bound_discovery" / "scripts"
        / "run_slack_discovery.py")
    config = frozen["cost14"]
    exact_cost = config["exact_cost"]

    started = time.perf_counter()
    layers, timings = pilot.exact_sparse_layers("AND_OR_NOT", 5, exact_cost)
    enumeration_seconds = time.perf_counter() - started
    exact_layer = np.asarray(sorted(layers[exact_cost]), dtype=np.uint64)
    rng = np.random.default_rng(config["selection_seed"])
    pool_size = min(config["candidate_pool_size"], len(exact_layer))
    pool_tables = np.sort(rng.choice(exact_layer, size=pool_size, replace=False))

    four_costs = pd.read_csv(FOUR / "feature_table.csv").target_AND_OR_NOT.to_numpy(int)
    sandwich = pilot.load_sandwich()
    catalog = sandwich.cube_catalog(5)
    rows = []
    for value in pool_tables:
        table = int(value)
        compact = pilot.compact_features(table, 5)
        cover = pilot.exact_aon_cover(table, 5, sandwich, catalog)
        existing_upper = float(pilot.universal_upper(
            "AND_OR_NOT",
            np.asarray([compact["decision_tree_leaf_count"]]),
            np.asarray([cover]))[0])
        structure = next_step.construction_features(table, 5, four_costs)
        rows.append({
            "truth_table": table, "inputs": 5,
            "exact_minimum_gates": exact_cost,
            "classical_lower_k_plus_1": compact["khrapchenko_product"],
            "existing_upper_k_plus_1": existing_upper,
            **compact, **structure,
        })
    pool = slack.add_bound_columns(pd.DataFrame(rows))
    prediction = np.maximum(0.0, teacher.predict(pool[frozen["features"]]))
    reduction, groups = apply_policy(prediction, frozen)
    pool["risk_stratum"] = groups
    pool["predicted_residual_slack"] = prediction
    pool["selective_reduction"] = reduction
    pool["selective_upper_k_plus_1"] = pool.construction_upper_k_plus_1 - reduction

    positive = pool[pool.selective_reduction > 0].sort_values(
        ["selective_reduction", "truth_table"], ascending=[False, True])
    if len(positive) < config["targeted_size"]:
        raise RuntimeError(
            f"only {len(positive)} positive reductions; {config['targeted_size']} required")
    targeted = positive.head(config["targeted_size"]).copy()
    remainder = pool.drop(index=targeted.index)
    reference_indices = rng.choice(
        remainder.index.to_numpy(), size=config["reference_size"], replace=False)
    reference = remainder.loc[reference_indices].copy()
    targeted["cohort"] = "targeted"
    reference["cohort"] = "random_reference"
    selected = pd.concat([targeted, reference], ignore_index=True)
    selected["covered"] = (
        selected.exact_k_plus_1 <= selected.selective_upper_k_plus_1 + 1e-10)
    selected["violation"] = np.maximum(
        0.0, selected.exact_k_plus_1 - selected.selective_upper_k_plus_1)

    summary = pd.DataFrame([{
        "cohort": cohort,
        "functions": len(group),
        "coverage": float(group.covered.mean()),
        "misses": int((~group.covered).sum()),
        "mean_construction_upper": float(group.construction_upper_k_plus_1.mean()),
        "mean_selective_upper": float(group.selective_upper_k_plus_1.mean()),
        "mean_reduction_beyond_construction": float(group.selective_reduction.mean()),
        "functions_improved": int(np.sum(group.selective_reduction > 0)),
        "maximum_violation": float(group.violation.max()),
    } for cohort, group in selected.groupby("cohort", sort=False)])

    coverage = float(selected.covered.mean())
    maximum_violation = float(selected.violation.max())
    mean_reduction = float(selected.selective_reduction.mean())
    improved_fraction = float(np.mean(selected.selective_reduction > 0))
    criteria = config["success_criteria"]
    passed = (
        coverage >= criteria["coverage"]
        and maximum_violation <= criteria["maximum_violation"]
        and (mean_reduction > 0 if criteria["positive_mean_reduction_beyond_construction"]
             else True)
        and improved_fraction >= criteria["minimum_improved_fraction"]
    )

    selected.to_csv(ARTIFACTS / "cost14_prospective_predictions.csv", index=False)
    summary.to_csv(ARTIFACTS / "cost14_prospective_summary.csv", index=False)
    analysis = {
        "frozen_protocol_sha256": frozen_hash,
        "enumeration": {
            "exact_layer_functions": len(exact_layer),
            "cumulative_functions": sum(len(layer) for layer in layers),
            "seconds": enumeration_seconds,
            "per_layer_seconds": timings,
        },
        "overall": {
            "coverage": coverage,
            "maximum_violation": maximum_violation,
            "mean_reduction_beyond_construction": mean_reduction,
            "improved_fraction": improved_fraction,
            "passed_frozen_success_criteria": passed,
        },
        "summary": summary.to_dict(orient="records"),
    }
    (ARTIFACTS / "cost14_prospective_analysis.json").write_text(
        json.dumps(analysis, indent=2), encoding="utf-8")
    lines = [
        "# Frozen prospective cost-14 selective result", "",
        f"Frozen protocol SHA-256: `{frozen_hash}`", "",
        f"Exact cost-14 layer: {len(exact_layer):,} functions; exhaustive enumeration",
        f"through cost 14: {sum(len(layer) for layer in layers):,} functions in",
        f"{enumeration_seconds:.1f} seconds.", "",
        summary.round(4).to_markdown(index=False), "",
        f"Overall coverage: {coverage:.4f}.",
        f"Maximum violation: {maximum_violation:.4f}.",
        f"Mean reduction beyond construction: {mean_reduction:.4f}.",
        f"Improved fraction: {improved_fraction:.4f}.",
        f"Passed frozen success criteria: `{passed}`.", "",
    ]
    (REPORTS / "COST14_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary.round(4).to_string(index=False))
    print(f"passed_frozen_success_criteria={passed}")


if __name__ == "__main__":
    main()
