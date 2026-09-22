"""Prospective five-input AON upper-tail test at exact formula cost 12."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parents[1]
ARTIFACTS = EXPERIMENT / "artifacts"
REPORTS = EXPERIMENT / "reports"
PILOT = ROOT / "experiments" / "five_bit_pilot"
FOUR_BIT = ROOT / "experiments" / "four_bit"
PROTOCOL_PATH = EXPERIMENT / "protocol.json"

BASE_FEATURES = [
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
STRUCTURE_FEATURES = [
    "and_factorable_partitions",
    "or_factorable_partitions",
    "canalizing_variable_count",
    "restriction_upper_k_plus_1",
    "factor_upper_k_plus_1",
    "construction_upper_k_plus_1",
    "prime_cover_slack",
]
MODEL_SPEC = {
    "loss": "quantile",
    "quantile": 0.90,
    "learning_rate": 0.06,
    "max_iter": 250,
    "max_leaf_nodes": 10,
    "min_samples_leaf": 15,
    "l2_regularization": 2.0,
    "early_stopping": False,
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compact_table(table: int, variables: list[int], n: int) -> int:
    result = 0
    for local_assignment in range(1 << len(variables)):
        assignment = sum(
            ((local_assignment >> index) & 1) << variable
            for index, variable in enumerate(variables)
        )
        result |= ((table >> assignment) & 1) << local_assignment
    return result


def expand_to_four(table: int, n: int) -> int:
    """Add unused high variables so a <=4-input function indexes n=4 data."""
    mask = (1 << n) - 1
    return sum(((table >> (assignment & mask)) & 1) << assignment
               for assignment in range(16))


def exact_subcost(table: int, n: int, aon_cost_four: np.ndarray) -> int:
    return int(aon_cost_four[expand_to_four(table, n)])


def construction_features(table: int, n: int,
                          aon_cost_four: np.ndarray) -> dict[str, float]:
    """Certified restriction/factor upper bounds and related structure."""
    full_mask = (1 << (1 << n)) - 1
    restriction_upper = float("inf")
    canalizing = 0
    for variable in range(n):
        cofactors = []
        for value in (0, 1):
            reduced = 0
            for local in range(1 << (n - 1)):
                low = local & ((1 << variable) - 1)
                assignment = low | ((local >> variable) << (variable + 1))
                assignment |= value << variable
                reduced |= ((table >> assignment) & 1) << local
            cofactors.append(reduced)
        cofactor_mask = (1 << (1 << (n - 1))) - 1
        if any(item in (0, cofactor_mask) for item in cofactors):
            canalizing += 1
        # K(f) <= K(f_0) + K(f_1) + 4, hence K(f)+1 below.
        restriction_upper = min(
            restriction_upper,
            exact_subcost(cofactors[0], n - 1, aon_cost_four)
            + exact_subcost(cofactors[1], n - 1, aon_cost_four) + 5,
        )

    and_count = 0
    or_count = 0
    factor_upper = float("inf")
    for left_mask in range(1, (1 << n) - 1):
        if not left_mask & 1:
            continue
        left_variables = [i for i in range(n) if left_mask & (1 << i)]
        right_variables = [i for i in range(n) if not left_mask & (1 << i)]
        rows = 1 << len(left_variables)
        columns = 1 << len(right_variables)
        matrix = np.empty((rows, columns), dtype=bool)
        for row in range(rows):
            for column in range(columns):
                assignment = sum(((row >> j) & 1) << variable
                                 for j, variable in enumerate(left_variables))
                assignment |= sum(((column >> j) & 1) << variable
                                  for j, variable in enumerate(right_variables))
                matrix[row, column] = bool((table >> assignment) & 1)

        row_one = np.any(matrix, axis=1)
        column_one = np.any(matrix, axis=0)
        if (0 < int(row_one.sum()) < rows
                and 0 < int(column_one.sum()) < columns
                and np.array_equal(matrix, row_one[:, None] & column_one[None, :])):
            and_count += 1
            left_table = sum(int(bit) << index for index, bit in enumerate(row_one))
            right_table = sum(int(bit) << index for index, bit in enumerate(column_one))
            # One AND combines the factors: K(f)+1=(K(g)+1)+(K(h)+1).
            factor_upper = min(
                factor_upper,
                exact_subcost(left_table, len(left_variables), aon_cost_four)
                + exact_subcost(right_table, len(right_variables), aon_cost_four) + 2,
            )

        zeros = ~matrix
        row_zero = np.any(zeros, axis=1)
        column_zero = np.any(zeros, axis=0)
        if (0 < int(row_zero.sum()) < rows
                and 0 < int(column_zero.sum()) < columns
                and np.array_equal(zeros, row_zero[:, None] & column_zero[None, :])):
            or_count += 1
            left_table = sum(int(not bit) << index for index, bit in enumerate(row_zero))
            right_table = sum(int(not bit) << index for index, bit in enumerate(column_zero))
            factor_upper = min(
                factor_upper,
                exact_subcost(left_table, len(left_variables), aon_cost_four)
                + exact_subcost(right_table, len(right_variables), aon_cost_four) + 2,
            )

    # Infinity is useful as a logical result but not as a model coordinate.
    absent_factor_bound = 3 + 6 * (2 ** n - 1)
    if not np.isfinite(factor_upper):
        factor_upper = float(absent_factor_bound)
    return {
        "and_factorable_partitions": float(and_count),
        "or_factorable_partitions": float(or_count),
        "canalizing_variable_count": float(canalizing),
        "restriction_upper_k_plus_1": float(restriction_upper),
        "factor_upper_k_plus_1": float(factor_upper),
    }


def add_structure(table: pd.DataFrame, aon_cost_four: np.ndarray) -> pd.DataFrame:
    rows = [construction_features(int(value), 5, aon_cost_four)
            for value in table.truth_table]
    result = table.reset_index(drop=True).copy()
    result = pd.concat([result, pd.DataFrame(rows)], axis=1)
    result["construction_upper_k_plus_1"] = np.minimum.reduce([
        result.classical_upper_k_plus_1.to_numpy(float),
        result.restriction_upper_k_plus_1.to_numpy(float),
        result.factor_upper_k_plus_1.to_numpy(float),
    ])
    result["prime_cover_slack"] = (
        result.classical_upper_k_plus_1 - result.predicted_k_plus_1)
    return result


def conditional_interval(train: pd.DataFrame, calibration: pd.DataFrame,
                         test: pd.DataFrame, features: list[str], seed: int,
                         prefix: str) -> pd.DataFrame:
    result = test.copy()
    exact_train = train.exact_minimum_gates.to_numpy(float) + 1
    exact_cal = calibration.exact_minimum_gates.to_numpy(float) + 1
    allowances = []
    for side_index, side in enumerate(("lower", "upper")):
        if side == "lower":
            train_score = np.maximum(0, train.predicted_k_plus_1 - exact_train)
            cal_score = np.maximum(0, calibration.predicted_k_plus_1 - exact_cal)
        else:
            train_score = np.maximum(0, exact_train - train.predicted_k_plus_1)
            cal_score = np.maximum(0, exact_cal - calibration.predicted_k_plus_1)
        model = HistGradientBoostingRegressor(
            **MODEL_SPEC, random_state=seed + side_index)
        model.fit(train[features], train_score)
        offset = float(np.max(cal_score - model.predict(calibration[features])))
        allowances.append(np.maximum(0, model.predict(test[features]) + offset))
    result[f"{prefix}_lower_k_plus_1"] = np.maximum(
        result.classical_lower_k_plus_1,
        result.predicted_k_plus_1 - allowances[0])
    result[f"{prefix}_upper_k_plus_1"] = np.minimum(
        result.classical_upper_k_plus_1,
        result.predicted_k_plus_1 + allowances[1])
    exact = result.exact_minimum_gates + 1
    result[f"{prefix}_covered"] = (
        (result[f"{prefix}_lower_k_plus_1"] <= exact)
        & (exact <= result[f"{prefix}_upper_k_plus_1"]))
    return result


def summarize(table: pd.DataFrame, prefix: str, cohort: str) -> dict[str, object]:
    group = table[table.cohort == cohort]
    learned_width = (group[f"{prefix}_upper_k_plus_1"]
                     - group[f"{prefix}_lower_k_plus_1"])
    classical_width = (group.classical_upper_k_plus_1
                       - group.classical_lower_k_plus_1)
    exact = group.exact_minimum_gates + 1
    return {
        "cohort": cohort,
        "method": prefix,
        "functions": len(group),
        "coverage": float(group[f"{prefix}_covered"].mean()),
        "misses_below": int(np.sum(exact < group[f"{prefix}_lower_k_plus_1"])),
        "misses_above": int(np.sum(exact > group[f"{prefix}_upper_k_plus_1"])),
        "mean_learned_width": float(learned_width.mean()),
        "mean_relative_shrinkage": float(np.mean(1 - learned_width / classical_width)),
    }


def main() -> None:
    protocol_bytes = PROTOCOL_PATH.read_bytes()
    protocol = json.loads(protocol_bytes)
    protocol_hash = hashlib.sha256(protocol_bytes).hexdigest()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    pilot = load_module("five_bit_pilot_core", PILOT / "scripts" / "run_local_pilot.py")
    core = pilot.load_core()
    sandwich = pilot.load_sandwich()
    four = pd.read_csv(FOUR_BIT / "artifacts" / "feature_table.csv")
    positions = pd.read_csv(FOUR_BIT / "artifacts" / "complexity_sandwich_positions.csv")
    four["exact_aon_cover_upper"] = positions.exact_aon_cover_upper
    aon_cost_four = four.target_AND_OR_NOT.to_numpy(int)

    historical = pd.read_csv(PILOT / "artifacts" / "five_bit_pilot_predictions.csv")
    historical = historical[historical.language == "AND_OR_NOT"].copy()
    historical["classical_width"] = (
        historical.classical_upper_k_plus_1 - historical.classical_lower_k_plus_1)
    historical = add_structure(historical, aon_cost_four)
    train = historical[historical.exact_minimum_gates <= protocol["training_max_cost"]]
    calibration = historical[
        historical.exact_minimum_gates == protocol["calibration_cost"]]

    started = time.perf_counter()
    layers, timings = pilot.exact_sparse_layers(
        "AND_OR_NOT", protocol["inputs"], protocol["test_cost"])
    enumeration_seconds = time.perf_counter() - started
    test_layer = np.asarray(sorted(layers[protocol["test_cost"]]), dtype=np.uint64)
    rng = np.random.default_rng(protocol["selection_seed"])
    pool_size = min(protocol["candidate_pool_size"], len(test_layer))
    pool_tables = np.sort(rng.choice(test_layer, size=pool_size, replace=False))

    model, _, _ = pilot.fit_frozen_universal_model(core, four, "AND_OR_NOT", 2)
    catalog = sandwich.cube_catalog(protocol["inputs"])
    feature_rows = []
    for value in pool_tables:
        table = int(value)
        features = pilot.compact_features(table, protocol["inputs"])
        features["exact_aon_cover_upper"] = pilot.exact_aon_cover(
            table, protocol["inputs"], sandwich, catalog)
        feature_rows.append(features)
    pool = pd.DataFrame(feature_rows)
    pool.insert(0, "truth_table", pool_tables.astype(np.uint64))
    matrix = pool[core.COMPACT].to_numpy(float)
    lower = pool.khrapchenko_product.to_numpy(float)
    upper = pilot.universal_upper(
        "AND_OR_NOT", pool.decision_tree_leaf_count.to_numpy(float),
        pool.exact_aon_cover_upper.to_numpy(float))
    predicted_position = np.clip(model.predict(matrix), 0, 1)
    pool["exact_minimum_gates"] = protocol["test_cost"]
    pool["classical_lower_k_plus_1"] = lower
    pool["classical_upper_k_plus_1"] = upper
    pool["predicted_k_plus_1"] = lower + predicted_position * (upper - lower)
    pool["classical_width"] = upper - lower
    pool = add_structure(pool, aon_cost_four)

    ranked = pool.sort_values(
        ["prime_cover_slack", "truth_table"], ascending=[False, True])
    upper_tail = ranked.head(protocol["upper_tail_size"]).copy()
    remainder = ranked.iloc[protocol["upper_tail_size"]:]
    reference_indices = rng.choice(
        remainder.index.to_numpy(), size=protocol["reference_size"], replace=False)
    reference = remainder.loc[reference_indices].copy()
    upper_tail["cohort"] = "upper_tail"
    reference["cohort"] = "random_reference"
    selected = pd.concat([upper_tail, reference], ignore_index=True)

    selected = conditional_interval(
        train, calibration, selected, BASE_FEATURES,
        protocol["conditional_model_seed"] + 200, "baseline")
    selected = conditional_interval(
        train, calibration, selected, BASE_FEATURES + STRUCTURE_FEATURES,
        protocol["conditional_model_seed"] + 400, "structure_augmented")

    summaries = []
    for cohort in ("upper_tail", "random_reference"):
        for method in ("baseline", "structure_augmented"):
            summaries.append(summarize(selected, method, cohort))
    summary = pd.DataFrame(summaries)
    construction_summary = pd.DataFrame([{
        "cohort": cohort,
        "functions": len(group),
        "functions_improved": int(np.sum(
            group.construction_upper_k_plus_1 < group.classical_upper_k_plus_1)),
        "restriction_improved": int(np.sum(
            group.restriction_upper_k_plus_1 < group.classical_upper_k_plus_1)),
        "factor_improved": int(np.sum(
            group.factor_upper_k_plus_1 < group.classical_upper_k_plus_1)),
        "mean_existing_upper": float(group.classical_upper_k_plus_1.mean()),
        "mean_construction_upper": float(group.construction_upper_k_plus_1.mean()),
        "mean_certified_improvement": float(np.mean(
            group.classical_upper_k_plus_1 - group.construction_upper_k_plus_1)),
    } for cohort, group in selected.groupby("cohort", sort=False)])

    selected.to_csv(ARTIFACTS / "aon_cost_12_selected.csv", index=False)
    summary.to_csv(ARTIFACTS / "aon_cost_12_envelope_summary.csv", index=False)
    construction_summary.to_csv(
        ARTIFACTS / "aon_cost_12_construction_summary.csv", index=False)
    metadata = {
        "protocol_sha256": protocol_hash,
        "protocol": protocol,
        "enumeration": {
            "layer_size": len(test_layer),
            "cumulative_functions": sum(len(layer) for layer in layers),
            "seconds": enumeration_seconds,
            "per_layer_seconds": timings,
        },
        "envelope_summary": summary.to_dict(orient="records"),
        "construction_summary": construction_summary.to_dict(orient="records"),
    }
    (ARTIFACTS / "aon_cost_12_analysis.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")

    display = summary.copy()
    for column in ("coverage", "mean_learned_width", "mean_relative_shrinkage"):
        display[column] = display[column].round(4)
    construction_display = construction_summary.copy()
    for column in ("mean_existing_upper", "mean_construction_upper",
                   "mean_certified_improvement"):
        construction_display[column] = construction_display[column].round(4)
    lines = [
        "# Five-input AON upper-tail challenge", "",
        "This experiment is separate from the learned-envelope paper and leaves",
        "the prospective cost-11 result unchanged. The protocol was written and hashed",
        "before exact cost-12 enumeration and selection did not use prediction error.", "",
        f"Protocol SHA-256: `{protocol_hash}`", "",
        f"Exact cost-12 layer: {len(test_layer):,} functions; exhaustive enumeration",
        f"through cost 12: {sum(len(layer) for layer in layers):,} functions in",
        f"{enumeration_seconds:.1f} seconds.", "",
        "## Envelope results", "", display.to_markdown(index=False), "",
        "The upper-tail cohort has the largest prime-cover slack in a seeded",
        "1,000-function candidate pool. The random-reference cohort is sampled",
        "from the remainder. These layer-balanced challenge results are diagnostic,",
        "not a coverage theorem for arbitrary five-input functions.", "",
        "The structure features make both cohorts slightly narrower, but they do",
        "not improve coverage: targeted coverage falls from 91.25% to 88.75%,",
        "while random-reference coverage remains 77.5%. The opposite miss directions",
        "across cohorts show that prime-cover-slack selection induces a strong shift",
        "in the frozen point prediction, not a uniformly harder upper residual.", "",
        "## Certified construction results", "",
        construction_display.to_markdown(index=False), "",
        "The construction endpoint is the minimum of the existing prime-cover/tree",
        "bound, exact four-input cofactor restriction decompositions, and disjoint",
        "AND/OR factorizations. Every reported improvement is a certified formula",
        "upper bound, independent of learned-envelope coverage. In this sample all",
        "improvements come from restriction decompositions; disjoint factorization",
        "does not beat the existing endpoint.", "",
        "## Reproduce", "", "```powershell",
        "python experiments/five_bit_next_step/scripts/run_aon_upper_tail.py",
        "```", "",
    ]
    (REPORTS / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(display.to_string(index=False))
    print(construction_display.to_string(index=False))


if __name__ == "__main__":
    main()
