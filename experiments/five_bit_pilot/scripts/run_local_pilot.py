"""Local exact five-input feasibility pilot for a frozen learned envelope.

This deliberately enumerates only formula functions reachable through eleven
gates. It is a compute pilot and an out-of-dimension stress test on easy
functions, not a representative five-input benchmark.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


ROOT = Path(__file__).resolve().parents[3]
PILOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PILOT / "artifacts"
REPORTS = PILOT / "reports"
FOUR_BIT = ROOT / "experiments" / "four_bit"
MAX_COST = 11
SAMPLES_PER_LAYER = 40
SEED = 20260921
N = 5
LANGUAGES = ["NAND", "NOR", "AND_OR_NOT"]


def load_core():
    path = FOUR_BIT / "scripts" / "analyze_learned_envelope.py"
    spec = importlib.util.spec_from_file_location("learned_envelope_core", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_sandwich():
    path = FOUR_BIT / "scripts" / "analyze_complexity_sandwich.py"
    spec = importlib.util.spec_from_file_location("complexity_sandwich_core", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def variable_tables(n: int) -> list[int]:
    return [sum(((assignment >> variable) & 1) << assignment
                for assignment in range(1 << n))
            for variable in range(n)]


def exact_sparse_layers(language: str, n: int, max_cost: int):
    """Enumerate every distinct function first reached through ``max_cost``."""
    mask = (1 << (1 << n)) - 1
    variables = set(variable_tables(n))
    seen = set(variables)
    layers = [variables]
    timings = [0.0]
    for cost in range(1, max_cost + 1):
        started = time.perf_counter()
        outputs: set[int] = set()
        if language == "AND_OR_NOT":
            outputs.update((~item) & mask for item in layers[cost - 1])
        for left_cost in range((cost - 1) // 2 + 1):
            right_cost = cost - 1 - left_cost
            left = layers[left_cost]
            right = layers[right_cost]
            for left_function in left:
                if language == "NAND":
                    outputs.update((~(left_function & item)) & mask
                                   for item in right)
                elif language == "NOR":
                    outputs.update((~(left_function | item)) & mask
                                   for item in right)
                elif language == "AND_OR_NOT":
                    outputs.update(left_function & item for item in right)
                    outputs.update(left_function | item for item in right)
                else:
                    raise ValueError(language)
        outputs.difference_update(seen)
        seen.update(outputs)
        layers.append(outputs)
        timings.append(time.perf_counter() - started)
    return layers, timings


def compact_cofactor(table: int, n: int, variable: int, value: int) -> int:
    result = 0
    for reduced in range(1 << (n - 1)):
        low = reduced & ((1 << variable) - 1)
        assignment = low | ((reduced >> variable) << (variable + 1))
        assignment |= value << variable
        result |= ((table >> assignment) & 1) << reduced
    return result


@lru_cache(maxsize=None)
def decision_tree_leaves(table: int, n: int) -> int:
    mask = (1 << (1 << n)) - 1
    if table in (0, mask):
        return 1
    return min(
        decision_tree_leaves(compact_cofactor(table, n, variable, 0), n - 1)
        + decision_tree_leaves(compact_cofactor(table, n, variable, 1), n - 1)
        for variable in range(n)
    )


def minimum_certificates(bits: np.ndarray, n: int) -> np.ndarray:
    assignments = np.arange(1 << n)
    result = np.full(1 << n, n, dtype=np.int8)
    unresolved = np.ones(1 << n, dtype=bool)
    for size in range(n + 1):
        for variables in itertools.combinations(range(n), size):
            mask = sum(1 << variable for variable in variables)
            for assignment in assignments[unresolved]:
                group = assignments[(assignments & mask) == (assignment & mask)]
                if np.all(bits[group] == bits[assignment]):
                    result[assignment] = size
                    unresolved[assignment] = False
        if not np.any(unresolved):
            break
    return result


def compact_features(table: int, n: int) -> dict[str, float]:
    assignments = np.arange(1 << n)
    bits = np.asarray([(table >> int(index)) & 1 for index in assignments], dtype=np.int8)
    signs = 2 * bits.astype(np.int64) - 1
    zeros = int(np.sum(bits == 0))
    ones = len(bits) - zeros
    edges = 0
    for variable in range(n):
        edges += int(np.sum(bits != bits[assignments ^ (1 << variable)]))
    edges //= 2
    boundary = 0.0 if zeros == 0 or ones == 0 else edges * edges / (zeros * ones)

    certificates = minimum_certificates(bits, n)
    output_means = []
    for output in (0, 1):
        selected = certificates[bits == output]
        output_means.append(float(np.mean(selected)) if len(selected) else 0.0)

    anf = bits.copy()
    for variable in range(n):
        for monomial in range(1 << n):
            if monomial & (1 << variable):
                anf[monomial] ^= anf[monomial ^ (1 << variable)]
    degrees = np.asarray([int(index).bit_count() for index in assignments])

    walsh = signs.copy()
    stride = 1
    while stride < len(walsh):
        for start in range(0, len(walsh), 2 * stride):
            left = walsh[start:start + stride].copy()
            right = walsh[start + stride:start + 2 * stride].copy()
            walsh[start:start + stride] = left + right
            walsh[start + stride:start + 2 * stride] = left - right
        stride *= 2

    result = {
        "khrapchenko_product": boundary,
        "decision_tree_leaf_count": float(decision_tree_leaves(table, n)),
        "mean_certificate_size": float(np.mean(certificates)),
        "maximum_certificate_size": float(np.max(certificates)),
        "mean_certificate_size_output_0": output_means[0],
        "mean_certificate_size_output_1": output_means[1],
    }
    for degree in range(4):
        result[f"anf_terms_degree_{degree}"] = float(
            np.sum(anf[degrees == degree]))
        result[f"fourier_negative_degree_{degree}"] = float(
            np.sum(walsh[degrees == degree] < 0))
    result["anf_terms_degree_4"] = float(np.sum(anf[degrees >= 4]))
    result["fourier_negative_degree_4"] = float(
        np.sum(walsh[degrees >= 4] < 0))
    return result


def universal_upper(language: str, leaves: np.ndarray,
                    cover: np.ndarray | None = None) -> np.ndarray:
    if language in ("NAND", "NOR"):
        return 6 + 9 * (leaves - 1)
    tree = 3 + 6 * (leaves - 1)
    return tree if cover is None else np.minimum(tree, cover)


def exact_aon_cover(table: int, n: int, sandwich, catalog) -> int:
    functions = 1 << (1 << n)
    full_mask = functions - 1
    if table in (0, full_mask):
        return 3
    zeros = full_mask ^ table
    return min(
        sandwich.minimum_prime_cover(table, catalog, "dnf_weight", full_mask),
        sandwich.minimum_prime_cover(zeros, catalog, "cnf_weight", full_mask),
        sandwich.minimum_prime_cover(table, catalog, "cnf_weight", full_mask) + 1,
        sandwich.minimum_prime_cover(zeros, catalog, "dnf_weight", full_mask) + 1,
    )


def fit_frozen_universal_model(core, four: pd.DataFrame, language: str,
                               language_index: int):
    values = four[core.COMPACT].to_numpy(float)
    unique, _, groups = np.unique(
        values, axis=0, return_index=True, return_inverse=True)
    exact = four[f"target_{language}"].to_numpy(float) + 1
    lower = four.khrapchenko_product.to_numpy(float)
    cover = (four.exact_aon_cover_upper.to_numpy(float)
             if language == "AND_OR_NOT" else None)
    upper = universal_upper(
        language, four.decision_tree_leaf_count.to_numpy(float), cover)
    width = upper - lower
    position = np.divide(
        exact - lower, width, out=np.full(len(exact), 0.5), where=width > 1e-12)
    weights, means = core.class_means(position, groups, len(unique))
    folds = list(KFold(n_splits=core.FOLDS, shuffle=True,
                       random_state=core.SEED).split(unique))
    class_prediction = np.empty(len(unique), dtype=float)
    for fold, (train, test) in enumerate(folds):
        model = core.make_model(language_index, fold, purpose=50)
        model.fit(unique[train], means[train], sample_weight=weights[train])
        class_prediction[test] = np.clip(model.predict(unique[test]), 0, 1)
    prediction = lower + class_prediction[groups] * width
    class_indices = np.arange(len(unique))
    minus, plus = core.class_worst_scores(
        prediction, exact, groups, class_indices)
    probability = 0.975
    allowance_minus = core.conformal_higher_quantile(minus, probability)
    allowance_plus = core.conformal_higher_quantile(plus, probability)

    frozen = core.make_model(language_index, 0, purpose=60)
    frozen.fit(unique, means, sample_weight=weights)
    return frozen, allowance_minus, allowance_plus


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    core = load_core()
    sandwich = load_sandwich()
    four = pd.read_csv(FOUR_BIT / "artifacts" / "feature_table.csv")
    four_positions = pd.read_csv(
        FOUR_BIT / "artifacts" / "complexity_sandwich_positions.csv")
    four["exact_aon_cover_upper"] = four_positions.exact_aon_cover_upper
    rng = np.random.default_rng(SEED)
    result_rows = []
    layer_rows = []

    for language_index, language in enumerate(LANGUAGES):
        layers, timings = exact_sparse_layers(language, N, MAX_COST)
        model, allowance_minus, allowance_plus = fit_frozen_universal_model(
            core, four, language, language_index)
        selected = []
        for cost, layer in enumerate(layers):
            ordered = np.asarray(sorted(layer), dtype=np.uint64)
            if len(ordered) > SAMPLES_PER_LAYER:
                ordered = np.sort(rng.choice(
                    ordered, size=SAMPLES_PER_LAYER, replace=False))
            selected.extend((int(table), cost) for table in ordered)
            layer_rows.append({
                "language": language,
                "cost": cost,
                "new_functions": len(layer),
                "cumulative_functions": sum(len(item) for item in layers[:cost + 1]),
                "seconds": timings[cost],
            })

        feature_rows = [compact_features(table, N) for table, _ in selected]
        if language == "AND_OR_NOT":
            catalog = sandwich.cube_catalog(N)
            for (table, _), feature_row in zip(selected, feature_rows):
                feature_row["exact_aon_cover_upper"] = exact_aon_cover(
                    table, N, sandwich, catalog)
        matrix = pd.DataFrame(feature_rows)[core.COMPACT].to_numpy(float)
        predicted_position = np.clip(model.predict(matrix), 0, 1)
        exact = np.asarray([cost for _, cost in selected], dtype=float) + 1
        lower = np.asarray([row["khrapchenko_product"] for row in feature_rows])
        leaves = np.asarray([row["decision_tree_leaf_count"] for row in feature_rows])
        cover = (np.asarray([row["exact_aon_cover_upper"] for row in feature_rows])
                 if language == "AND_OR_NOT" else None)
        upper = universal_upper(language, leaves, cover)
        prediction = lower + predicted_position * (upper - lower)
        learned_lower = np.maximum(lower, prediction - allowance_minus)
        learned_upper = np.minimum(upper, prediction + allowance_plus)
        for index, ((table, cost), features) in enumerate(zip(selected, feature_rows)):
            result_rows.append({
                "language": language,
                "truth_table": table,
                "exact_minimum_gates": cost,
                "classical_lower_k_plus_1": lower[index],
                "classical_upper_k_plus_1": upper[index],
                "predicted_k_plus_1": prediction[index],
                "calibrated_lower_k_plus_1": learned_lower[index],
                "calibrated_upper_k_plus_1": learned_upper[index],
                "covered": learned_lower[index] <= exact[index] <= learned_upper[index],
                "n4_lower_allowance": allowance_minus,
                "n4_upper_allowance": allowance_plus,
                **features,
            })

    results = pd.DataFrame(result_rows)
    layers = pd.DataFrame(layer_rows)
    results.to_csv(ARTIFACTS / "five_bit_pilot_predictions.csv", index=False)
    layers.to_csv(ARTIFACTS / "five_bit_sparse_layers.csv", index=False)
    summaries = []
    for language, group in results.groupby("language", sort=False):
        classical_width = (group.classical_upper_k_plus_1
                           - group.classical_lower_k_plus_1)
        learned_width = (group.calibrated_upper_k_plus_1
                         - group.calibrated_lower_k_plus_1)
        exact = group.exact_minimum_gates + 1
        summaries.append({
            "language": language,
            "sampled_functions": len(group),
            "maximum_exact_cost": int(group.exact_minimum_gates.max()),
            "coverage": float(group.covered.mean()),
            "mean_exact_minus_prediction": float(
                np.mean(exact - group.predicted_k_plus_1)),
            "misses_below_interval": int(np.sum(
                exact < group.calibrated_lower_k_plus_1)),
            "misses_above_interval": int(np.sum(
                exact > group.calibrated_upper_k_plus_1)),
            "mean_classical_width": float(classical_width.mean()),
            "mean_learned_width": float(learned_width.mean()),
            "mean_relative_shrinkage": float(
                np.mean(1 - learned_width / classical_width)),
            "enumerated_functions_through_max_cost": int(
                layers[layers.language == language].new_functions.sum()),
            "enumeration_seconds": float(
                layers[layers.language == language].seconds.sum()),
        })
    summary = pd.DataFrame(summaries)
    summary.to_csv(ARTIFACTS / "five_bit_pilot_summary.csv", index=False)
    metadata = {
        "status": "feasibility pilot; not representative of five-input functions",
        "inputs": N,
        "maximum_exact_cost": MAX_COST,
        "samples_per_cost_layer": SAMPLES_PER_LAYER,
        "seed": SEED,
        "degree_bins": ["0", "1", "2", "3", "4+"],
        "upper_endpoint": "dimension-independent decision-tree construction; exact prime cover also used for AND/OR/NOT",
        "summary": summary.to_dict(orient="records"),
    }
    (ARTIFACTS / "five_bit_pilot_analysis.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")

    display = summary.copy()
    for column in display.columns[2:]:
        display[column] = display[column].astype(float).round(4)
    lines = [
        "# Local five-input learned-envelope pilot", "",
        "This is a feasibility run over sampled functions whose exact minimum",
        "formula cost is at most eleven. Exactness follows from exhaustive sparse",
        "enumeration of every function reachable at each gate cost through eleven.",
        "It is strongly biased toward easy functions and is not a representative",
        "five-input validation set.", "", display.to_markdown(index=False), "",
        "The predictor is refit on four-input data using the general decision-tree",
        "upper bound, tightened by the exact prime-cover construction for",
        "AND/OR/NOT. Degree coordinates use fixed bins 0, 1, 2, 3, and 4+.", "",
        "Coverage here is diagnostic only. A decisive transfer experiment requires",
        "exact targets beyond cost eleven, selected before observing errors.", "",
        "## Reproduce", "", "```powershell",
        "python experiments/five_bit_pilot/scripts/run_local_pilot.py",
        "```", "",
    ]
    (REPORTS / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
