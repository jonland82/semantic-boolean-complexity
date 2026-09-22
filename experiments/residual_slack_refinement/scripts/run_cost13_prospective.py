"""Run the frozen prospective AON cost-13 residual-slack evaluation."""

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
FROZEN_PATH = ARTIFACTS / "frozen_cost13_protocol.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    frozen_hash = sha256(FROZEN_PATH)
    frozen = json.loads(FROZEN_PATH.read_text())
    observed_runtime = {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }
    if observed_runtime != frozen["runtime"]:
        raise RuntimeError(
            f"runtime mismatch: expected {frozen['runtime']}, observed {observed_runtime}")
    teacher_path = ARTIFACTS / frozen["teacher_artifact"]
    if sha256(teacher_path) != frozen["teacher_sha256"]:
        raise RuntimeError("frozen teacher artifact hash mismatch")

    pilot = load_module(
        "pilot_for_cost13",
        ROOT / "experiments" / "five_bit_pilot" / "scripts" / "run_local_pilot.py")
    next_step = load_module(
        "next_step_for_cost13",
        ROOT / "experiments" / "five_bit_next_step" / "scripts"
        / "run_aon_upper_tail.py")
    slack = load_module(
        "slack_for_cost13",
        ROOT / "experiments" / "slack_bound_discovery" / "scripts"
        / "run_slack_discovery.py")
    teacher = joblib.load(teacher_path)
    config = frozen["cost13"]
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
            "truth_table": table,
            "inputs": 5,
            "exact_minimum_gates": exact_cost,
            "classical_lower_k_plus_1": compact["khrapchenko_product"],
            "existing_upper_k_plus_1": existing_upper,
            **compact,
            **structure,
        })
    pool = slack.add_bound_columns(pd.DataFrame(rows))
    prediction = np.maximum(0.0, teacher.predict(pool[frozen["features"]]))
    reduction = np.maximum(0.0, prediction - frozen["teacher_cost12_offset"])
    pool["predicted_residual_slack"] = prediction
    pool["learned_reduction_beyond_construction"] = reduction
    pool["refined_upper_k_plus_1"] = pool.construction_upper_k_plus_1 - reduction

    ranked = pool.sort_values(
        ["learned_reduction_beyond_construction", "truth_table"],
        ascending=[False, True])
    targeted = ranked.head(config["targeted_size"]).copy()
    remainder = ranked.iloc[config["targeted_size"]:]
    reference_indices = rng.choice(
        remainder.index.to_numpy(), size=config["reference_size"], replace=False)
    reference = remainder.loc[reference_indices].copy()
    targeted["cohort"] = "targeted"
    reference["cohort"] = "random_reference"
    selected = pd.concat([targeted, reference], ignore_index=True)
    selected["covered"] = (
        selected.exact_k_plus_1 <= selected.refined_upper_k_plus_1 + 1e-10)
    selected["violation"] = np.maximum(
        0.0, selected.exact_k_plus_1 - selected.refined_upper_k_plus_1)

    rows = []
    for cohort, group in selected.groupby("cohort", sort=False):
        rows.append({
            "cohort": cohort,
            "functions": len(group),
            "coverage": float(group.covered.mean()),
            "misses": int((~group.covered).sum()),
            "mean_construction_upper": float(
                group.construction_upper_k_plus_1.mean()),
            "mean_refined_upper": float(group.refined_upper_k_plus_1.mean()),
            "mean_reduction_beyond_construction": float(
                group.learned_reduction_beyond_construction.mean()),
            "functions_improved": int(np.sum(
                group.learned_reduction_beyond_construction > 1e-12)),
            "maximum_violation": float(group.violation.max()),
        })
    summary = pd.DataFrame(rows)
    overall_coverage = float(selected.covered.mean())
    maximum_violation = float(selected.violation.max())
    mean_reduction = float(selected.learned_reduction_beyond_construction.mean())
    criteria = config["success_criteria"]
    passed = (
        overall_coverage >= criteria["coverage"]
        and maximum_violation <= criteria["maximum_violation"]
        and (mean_reduction > 0 if criteria["positive_mean_reduction_beyond_construction"]
             else True)
    )

    selected.to_csv(ARTIFACTS / "cost13_prospective_predictions.csv", index=False)
    summary.to_csv(ARTIFACTS / "cost13_prospective_summary.csv", index=False)
    analysis = {
        "frozen_protocol_sha256": frozen_hash,
        "enumeration": {
            "exact_layer_functions": len(exact_layer),
            "cumulative_functions": sum(len(layer) for layer in layers),
            "seconds": enumeration_seconds,
            "per_layer_seconds": timings,
        },
        "overall": {
            "coverage": overall_coverage,
            "maximum_violation": maximum_violation,
            "mean_reduction_beyond_construction": mean_reduction,
            "passed_frozen_success_criteria": passed,
        },
        "summary": summary.to_dict(orient="records"),
    }
    (ARTIFACTS / "cost13_prospective_analysis.json").write_text(
        json.dumps(analysis, indent=2), encoding="utf-8")
    display = summary.round(4)
    lines = [
        "# Frozen prospective cost-13 result", "",
        f"Frozen protocol SHA-256: `{frozen_hash}`", "",
        f"Exact cost-13 layer: {len(exact_layer):,} functions; exhaustive enumeration",
        f"through cost 13: {sum(len(layer) for layer in layers):,} functions in",
        f"{enumeration_seconds:.1f} seconds.", "",
        display.to_markdown(index=False), "",
        f"Overall coverage: {overall_coverage:.4f}.",
        f"Maximum violation: {maximum_violation:.4f}.",
        f"Mean reduction beyond construction: {mean_reduction:.4f}.",
        f"Passed frozen success criteria: `{passed}`.", "",
    ]
    (REPORTS / "COST13_RESULTS.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print(display.to_string(index=False))
    print(f"passed_frozen_success_criteria={passed}")


if __name__ == "__main__":
    main()
