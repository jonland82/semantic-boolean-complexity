"""Analyze lower/upper semantic envelopes around exact formula complexity."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold


EXPERIMENT = Path(__file__).resolve().parents[1]
ARTIFACTS = EXPERIMENT / "artifacts"
FIGURES = EXPERIMENT / "figures"
REPORTS = EXPERIMENT / "reports"
LANGUAGES = ["NAND", "NOR", "AND_OR_NOT"]
LABELS = {"NAND": "NAND", "NOR": "NOR", "AND_OR_NOT": "AND/OR/NOT"}
COLORS = {"lower": "#87939C", "exact": "#25292C", "upper": "#B98249"}
FOLDS = 5
SEED = 0

CERTIFICATES = [
    "mean_certificate_size",
    "maximum_certificate_size",
    "mean_certificate_size_output_0",
    "mean_certificate_size_output_1",
]
ALGEBRAIC_PHASE = (
    [f"anf_terms_degree_{degree}" for degree in range(5)]
    + [f"fourier_negative_degree_{degree}" for degree in range(5)]
)
CORRECTIONS = CERTIFICATES + ALGEBRAIC_PHASE
COMPACT = ["khrapchenko_product", "decision_tree_leaf_count"] + CORRECTIONS
FEATURE_SETS = {
    "bracket coordinates": ["khrapchenko_product", "decision_tree_leaf_count"],
    "certificates": CERTIFICATES,
    "algebraic + phase": ALGEBRAIC_PHASE,
    "all corrections": CORRECTIONS,
    "compact sandwich": COMPACT,
}


def cube_catalog(n: int) -> list[dict[str, object]]:
    """Enumerate cubes and their exact AON DNF/CNF ``K + 1`` weights."""
    assignments = range(1 << n)
    cubes = list(itertools.product((-1, 0, 1), repeat=n))
    index = {cube: cube_index for cube_index, cube in enumerate(cubes)}
    catalog = []
    for cube in cubes:
        coverage = 0
        for assignment in assignments:
            if all(value == -1 or ((assignment >> variable) & 1) == value
                   for variable, value in enumerate(cube)):
                coverage |= 1 << assignment
        literals = sum(value != -1 for value in cube)
        zeros = sum(value == 0 for value in cube)
        ones = sum(value == 1 for value in cube)
        parents = []
        for variable, value in enumerate(cube):
            if value != -1:
                parent = list(cube)
                parent[variable] = -1
                parents.append(index[tuple(parent)])
        catalog.append({
            "cube": cube,
            "coverage": coverage,
            "literals": literals,
            # For a DNF cube, fixed zeros require NOT gates.  For the CNF
            # clause complementary to a zero-cube, fixed ones require them.
            "dnf_weight": literals + zeros,
            "cnf_weight": literals + ones,
            "parents": parents,
        })
    return catalog


def minimum_prime_cover(target: int, catalog: list[dict[str, object]],
                        weight_name: str, full_mask: int,
                        return_cubes: bool = False):
    """Return the exact minimum prime-cube construction cost in ``K + 1`` units."""
    if target in (0, full_mask):
        return (3, []) if return_cubes else 3

    valid = [not (int(item["coverage"]) & ~target) for item in catalog]
    prime_indices = [
        index for index, item in enumerate(catalog)
        if valid[index] and not any(valid[parent] for parent in item["parents"])
    ]
    by_assignment = [[] for _ in range(full_mask.bit_length())]
    for index in prime_indices:
        coverage = int(catalog[index]["coverage"])
        for assignment in range(len(by_assignment)):
            if coverage & (1 << assignment):
                by_assignment[assignment].append(index)

    memo: dict[int, tuple[int, tuple[int, ...]]] = {0: (0, ())}

    def solve(uncovered: int) -> tuple[int, tuple[int, ...]]:
        if uncovered in memo:
            return memo[uncovered]
        first = (uncovered & -uncovered).bit_length() - 1
        best_cost = math.inf
        best_choice: tuple[int, ...] = ()
        for index in by_assignment[first]:
            item = catalog[index]
            remainder = uncovered & ~int(item["coverage"])
            remainder_cost, remainder_choice = solve(remainder)
            candidate = int(item[weight_name]) + remainder_cost
            if candidate < best_cost:
                best_cost = candidate
                best_choice = (index, *remainder_choice)
        if math.isinf(best_cost):
            raise AssertionError(f"no cube cover for target {target:#x}")
        memo[uncovered] = int(best_cost), best_choice
        return memo[uncovered]

    cost, choices = solve(target)
    return (cost, [catalog[index] for index in choices]) if return_cubes else cost


def exact_aon_cover_upper(n: int = 4) -> np.ndarray:
    """Compute the cheapest prime-cover construction, including an outer NOT."""
    assignments = 1 << n
    functions = 1 << assignments
    full_mask = functions - 1
    catalog = cube_catalog(n)
    candidates = np.empty((4, functions), dtype=np.int16)
    for table in range(functions):
        zeros = full_mask ^ table
        candidates[0, table] = minimum_prime_cover(
            table, catalog, "dnf_weight", full_mask)
        candidates[1, table] = minimum_prime_cover(
            zeros, catalog, "cnf_weight", full_mask)
        # Keeping a single outer NOT can be cheaper than pushing it through
        # the cover, because formulas do not share literal negations.
        candidates[2, table] = minimum_prime_cover(
            table, catalog, "cnf_weight", full_mask) + 1
        candidates[3, table] = minimum_prime_cover(
            zeros, catalog, "dnf_weight", full_mask) + 1
    candidates[:, [0, full_mask]] = 3
    return np.min(candidates, axis=0)


def aon_cover_details(table: int, n: int = 4) -> dict[str, object]:
    """Describe one minimum prime-cover construction for a truth table."""
    functions = 1 << (1 << n)
    full_mask = functions - 1
    if table in (0, full_mask):
        return {"cost": 3, "construction": "constant",
                "tied_constructions": "constant", "cubes": ""}
    catalog = cube_catalog(n)
    specs = [
        ("DNF(f)", table, "dnf_weight", 0),
        ("CNF(f)", full_mask ^ table, "cnf_weight", 0),
        ("NOT-CNF(not f)", table, "cnf_weight", 1),
        ("NOT-DNF(not f)", full_mask ^ table, "dnf_weight", 1),
    ]
    solved = []
    for name, target, weight_name, extra in specs:
        cost, cubes = minimum_prime_cover(
            target, catalog, weight_name, full_mask, return_cubes=True)
        solved.append((cost + extra, name, cubes))
    cost, construction, cubes = min(solved, key=lambda item: (item[0], item[1]))
    patterns = ["".join("-" if value == -1 else str(value)
                        for value in item["cube"]) for item in cubes]
    tied = [name for candidate, name, _ in solved if candidate == cost]
    return {
        "cost": cost,
        "construction": construction,
        "tied_constructions": ";".join(tied),
        "cubes": " ".join(patterns),
    }


def phase_permutation_canonical(table: int, n: int = 4) -> int:
    """Canonical truth table under input permutations and output complement."""
    assignments = range(1 << n)
    full_mask = (1 << (1 << n)) - 1
    best = full_mask
    for permutation in itertools.permutations(range(n)):
        transformed = 0
        for assignment in assignments:
            mapped = sum(((assignment >> permutation[index]) & 1) << index
                         for index in range(n))
            transformed |= ((table >> mapped) & 1) << assignment
        best = min(best, transformed, full_mask ^ transformed)
    return best


def anf_expression(table: int, n: int = 4) -> str:
    """Return a compact algebraic-normal-form expression."""
    coefficients = [(table >> assignment) & 1 for assignment in range(1 << n)]
    for variable in range(n):
        for monomial in range(1 << n):
            if monomial & (1 << variable):
                coefficients[monomial] ^= coefficients[monomial ^ (1 << variable)]
    terms = []
    for monomial, present in enumerate(coefficients):
        if not present:
            continue
        terms.append("1" if monomial == 0 else "*".join(
            f"x{variable}" for variable in range(n) if monomial & (1 << variable)))
    return " + ".join(terms) if terms else "0"


def contact_frame(table: pd.DataFrame, constants: pd.DataFrame,
                  lowers: dict[str, np.ndarray], uppers: dict[str, np.ndarray],
                  exact_cover: np.ndarray) -> pd.DataFrame:
    """Classify every function attaining a lower or upper envelope."""
    rows = []
    boundary = table["khrapchenko_product"].to_numpy(float)
    leaves = table["decision_tree_leaf_count"].to_numpy(int)
    for language in LANGUAGES:
        target = table[f"target_{language}"].to_numpy(int) + 1
        lower_hit = np.isclose(target, lowers[language])
        upper_hit = np.isclose(target, uppers[language])
        parameters = constants[constants.language == language].iloc[0]
        affine = (parameters.upper_anchor
                  + parameters.upper_slope * (leaves - 1))
        for function in np.flatnonzero(lower_hit | upper_hit):
            if lower_hit[function] and upper_hit[function]:
                contact = "both"
            elif lower_hit[function]:
                contact = "lower"
            else:
                contact = "upper"
            source = "empirical decision-tree envelope"
            cover = {"cost": "", "construction": "",
                     "tied_constructions": "", "cubes": ""}
            if language == "AND_OR_NOT":
                cover_active = np.isclose(uppers[language][function], exact_cover[function])
                affine_active = np.isclose(uppers[language][function], affine[function])
                source = "+".join(name for name, active in (
                    ("exact prime cover", cover_active),
                    ("empirical decision-tree envelope", affine_active)) if active)
                if cover_active:
                    cover = aon_cover_details(int(function))
            rows.append({
                "language": language,
                "contact": contact,
                "truth_table": int(function),
                "truth_table_hex": f"0x{function:04x}",
                "phase_permutation_orbit_hex":
                    f"0x{phase_permutation_canonical(int(function)):04x}",
                "anf": anf_expression(int(function)),
                "minimum_gates": int(target[function] - 1),
                "khrapchenko_product": boundary[function],
                "decision_tree_leaves": int(leaves[function]),
                "lower": lowers[language][function],
                "upper": uppers[language][function],
                "active_upper_source": source,
                "cover_cost_k_plus_1": cover["cost"],
                "cover_construction": cover["construction"],
                "tied_cover_constructions": cover["tied_constructions"],
                "cover_cubes_01_dash": cover["cubes"],
            })
    return pd.DataFrame(rows)


def class_moments(target: np.ndarray, groups: np.ndarray, classes: int):
    weights = np.bincount(groups, minlength=classes).astype(float)
    sums = np.bincount(groups, weights=target, minlength=classes)
    square_sums = np.bincount(groups, weights=target ** 2, minlength=classes)
    means = sums / weights
    within_sse = square_sums - sums ** 2 / weights
    return weights, means, within_sse


def weighted_r2(prediction: np.ndarray, weights: np.ndarray,
                means: np.ndarray, within_sse: np.ndarray) -> float:
    overall_mean = float(np.average(means, weights=weights))
    residual = float(np.sum(within_sse + weights * (means - prediction) ** 2))
    total = float(np.sum(within_sse + weights * (means - overall_mean) ** 2))
    return 1 - residual / total


def envelope(target: np.ndarray, boundary: np.ndarray, leaves: np.ndarray,
             upper_candidate: np.ndarray | None = None):
    positive = boundary > 0
    lower_scale = float(np.min(target[positive] / boundary[positive]))
    constant = leaves == 1
    upper_anchor = float(np.max(target[constant]))
    nonconstant = leaves > 1
    upper_slope = float(max(
        0.0, np.max((target[nonconstant] - upper_anchor) / (leaves[nonconstant] - 1))))
    lower = lower_scale * boundary
    upper = upper_anchor + upper_slope * (leaves - 1)
    if upper_candidate is not None:
        upper = np.minimum(upper, upper_candidate)
    if np.any(lower > target + 1e-10) or np.any(upper < target - 1e-10):
        raise AssertionError("computed envelope does not contain every exact target")
    width = upper - lower
    position = np.divide(target - lower, width, out=np.full_like(target, 0.5),
                         where=width > 1e-12)
    return lower_scale, upper_anchor, upper_slope, lower, upper, position


def binned_curves(target: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                  bins: int = 128):
    order = np.argsort(target, kind="stable")
    chunks = np.array_split(order, bins)
    percentile = 100 * (np.arange(len(chunks)) + 0.5) / len(chunks)
    summarize = lambda values: np.array([np.median(values[chunk]) for chunk in chunks])
    return percentile, summarize(lower), summarize(target), summarize(upper)


def fit_residual_models(table: pd.DataFrame, positions: dict[str, np.ndarray],
                        lowers: dict[str, np.ndarray], uppers: dict[str, np.ndarray]):
    compact_values = table[COMPACT].to_numpy(float)
    unique_compact, representative, groups = np.unique(
        compact_values, axis=0, return_index=True, return_inverse=True)
    classes = len(unique_compact)
    compact_index = {name: COMPACT.index(name) for name in COMPACT}
    folds = list(KFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(unique_compact))
    rows = []

    for language_index, language in enumerate(LANGUAGES):
        gate_target = table[f"target_{language}"].to_numpy(float) + 1
        position = positions[language]
        position_weights, position_means, position_within = class_moments(
            position, groups, classes)
        gate_weights, gate_means, gate_within = class_moments(gate_target, groups, classes)
        class_lower = lowers[language][representative]
        class_width = uppers[language][representative] - class_lower

        for fold, (train, test) in enumerate(folds):
            midpoint_position = np.full(len(test), 0.5)
            midpoint_gate = class_lower[test] + midpoint_position * class_width[test]
            rows.append({
                "language": language,
                "fold": fold,
                "model": "fixed midpoint",
                "features": 0,
                "position_r2": weighted_r2(midpoint_position, position_weights[test],
                                           position_means[test], position_within[test]),
                "gate_count_r2": weighted_r2(midpoint_gate, gate_weights[test],
                                             gate_means[test], gate_within[test]),
                "test_compact_classes": len(test),
            })
            for model_index, (model_name, feature_names) in enumerate(FEATURE_SETS.items()):
                columns = [compact_index[name] for name in feature_names]
                model = HistGradientBoostingRegressor(
                    learning_rate=0.06,
                    max_iter=250,
                    max_leaf_nodes=15,
                    min_samples_leaf=10,
                    l2_regularization=1.0,
                    early_stopping=False,
                    random_state=SEED + 100 * language_index + 10 * model_index + fold,
                )
                model.fit(unique_compact[train][:, columns], position_means[train],
                          sample_weight=position_weights[train])
                predicted_position = np.clip(model.predict(unique_compact[test][:, columns]), 0, 1)
                predicted_gate = class_lower[test] + predicted_position * class_width[test]
                rows.append({
                    "language": language,
                    "fold": fold,
                    "model": model_name,
                    "features": len(columns),
                    "position_r2": weighted_r2(
                        predicted_position, position_weights[test], position_means[test],
                        position_within[test]),
                    "gate_count_r2": weighted_r2(
                        predicted_gate, gate_weights[test], gate_means[test], gate_within[test]),
                    "test_compact_classes": len(test),
                })
    return pd.DataFrame(rows), classes


def plot_sandwich(table: pd.DataFrame, lowers: dict[str, np.ndarray],
                  uppers: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), sharey=False)
    for axis, language in zip(axes, LANGUAGES):
        target = table[f"target_{language}"].to_numpy(float) + 1
        x, lower, exact, upper = binned_curves(target, lowers[language], uppers[language])
        axis.fill_between(x, lower, upper, color="#D8C1A8", alpha=0.42, linewidth=0)
        axis.plot(x, lower, color=COLORS["lower"], linewidth=1.8,
                  label=r"Khrapchenko lower $cB$")
        axis.plot(x, exact, color=COLORS["exact"], linewidth=2.2,
                  label=r"exact $K_{\mathcal{L}}+1$")
        axis.plot(x, upper, color=COLORS["upper"], linewidth=1.8,
                  label=r"tree upper $A+C(U-1)$")
        axis.set_title(LABELS[language], loc="left", fontweight="bold")
        axis.set_xlabel("functions sorted by exact complexity (percentile)")
        axis.grid(axis="y", color="#D9DDDF", linewidth=0.7)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("leaf/gate-comparable cost")
    axes[-1].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Exact formula complexity inside empirical semantic envelopes",
                 x=0.06, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for extension in ("png", "svg", "pdf"):
        fig.savefig(FIGURES / f"complexity_sandwich.{extension}", dpi=220,
                    bbox_inches="tight")
    plt.close(fig)


def plot_sandwich_v2(table: pd.DataFrame, lowers: dict[str, np.ndarray],
                     uppers: dict[str, np.ndarray], exact_cover: np.ndarray,
                     constants: pd.DataFrame,
                     stem: str = "complexity_sandwich_v2",
                     title: str = ("Complexity sandwich v2: theorem-valid lower bound "
                                   "and sharpened constructions")) -> None:
    """Plot the sharp four-bit envelopes and identify their theoretical status."""
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.8), sharey=False)
    leaves = table["decision_tree_leaf_count"].to_numpy(float)
    annotations = {
        "NAND": (r"sharp $n=4$: $6+\frac{9}{4}(U-1)$",
                 r"all $n$: $6+9(U-1)$"),
        "NOR": (r"sharp $n=4$: $6+\frac{9}{4}(U-1)$",
                r"all $n$: $6+9(U-1)$"),
        "AND_OR_NOT": (r"sharp $n=4$: $\min\{3+\frac{13}{9}(U-1),Q_{\min}\}$",
                       r"all $n$: $3+6(U-1)$"),
    }
    handles = {}
    for axis, language in zip(axes, LANGUAGES):
        target = table[f"target_{language}"].to_numpy(float) + 1
        x, lower, exact, upper = binned_curves(
            target, lowers[language], uppers[language])
        band = axis.fill_between(
            x, lower, upper, color="#D8C1A8", alpha=0.38, linewidth=0,
            label="verified four-bit interval")
        lower_line, = axis.plot(
            x, lower, color=COLORS["lower"], linewidth=1.9,
            label=r"general lower $B(f)$")
        exact_line, = axis.plot(
            x, exact, color=COLORS["exact"], linewidth=2.4,
            label=r"exact $K_{\mathcal{L}}(f)+1$")
        upper_line, = axis.plot(
            x, upper, color=COLORS["upper"], linewidth=2.1,
            label=r"sharp $n=4$ upper")
        handles.update({"band": band, "lower": lower_line,
                        "exact": exact_line, "upper": upper_line})

        if language == "AND_OR_NOT":
            row = constants[constants.language == language].iloc[0]
            tree_upper = row.upper_anchor + row.upper_slope * (leaves - 1)
            _, _, _, tree_curve = binned_curves(
                target, lowers[language], tree_upper)
            _, _, _, cover_curve = binned_curves(
                target, lowers[language], exact_cover)
            tree_line, = axis.plot(
                x, tree_curve, color="#B98249", linewidth=1.25,
                linestyle="--", alpha=0.8, label="tree component")
            cover_line, = axis.plot(
                x, cover_curve, color="#6F7F71", linewidth=1.25,
                linestyle=":", alpha=0.95, label=r"prime-cover $Q_{\min}$")
            handles.update({"tree": tree_line, "cover": cover_line})

        sharp, general = annotations[language]
        axis.text(
            0.035, 0.955, f"{sharp}\n{general}", transform=axis.transAxes,
            ha="left", va="top", fontsize=8.2, linespacing=1.35,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white",
                  "edgecolor": "#D9DDDF", "alpha": 0.92})
        axis.set_title(LABELS[language], loc="left", fontweight="bold", pad=9)
        axis.set_xlabel("functions sorted by exact complexity (percentile)")
        axis.grid(axis="y", color="#D9DDDF", linewidth=0.7)
        axis.set_axisbelow(True)

    axes[0].set_ylabel(r"cost in $K+1$ / formula-leaf units")
    legend_order = ["lower", "exact", "upper", "tree", "cover", "band"]
    fig.legend(
        [handles[name] for name in legend_order if name in handles],
        [handles[name].get_label() for name in legend_order if name in handles],
        loc="upper center", bbox_to_anchor=(0.5, 0.895), ncol=6,
        frameon=False, fontsize=8.3, columnspacing=1.4, handlelength=2.5)
    fig.suptitle(
        title,
        x=0.055, y=0.995, ha="left", fontsize=14, fontweight="bold")
    fig.text(
        0.055, 0.948,
        "Solid upper curves are exhaustive four-input envelopes; the second formula in each panel holds for every input dimension.",
        ha="left", va="top", fontsize=9, color="#4A5054")
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.80), w_pad=2.0)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(FIGURES / f"{stem}.{extension}", dpi=220,
                    bbox_inches="tight")
    plt.close(fig)


def plot_positions(table: pd.DataFrame, positions: dict[str, np.ndarray]) -> None:
    certificate = table["mean_certificate_size"].to_numpy(float)
    anf_linear = table["anf_terms_degree_1"].to_numpy(float)
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.0), sharey=True)
    top_image = None
    bottom_image = None
    for column, language in enumerate(LANGUAGES):
        target = table[f"target_{language}"].to_numpy(float) + 1
        position = positions[language]
        top_image = axes[0, column].hexbin(
            target, position, C=certificate, gridsize=(24, 28), mincnt=1,
            reduce_C_function=np.mean, cmap="viridis", vmin=certificate.min(),
            vmax=certificate.max())
        bottom_image = axes[1, column].hexbin(
            target, position, C=anf_linear, gridsize=(24, 28), mincnt=1,
            reduce_C_function=np.mean, cmap="cividis", vmin=anf_linear.min(),
            vmax=anf_linear.max())
        axes[0, column].set_title(LABELS[language], loc="left", fontweight="bold")
        axes[1, column].set_xlabel(r"exact $K_{\mathcal{L}}(f)+1$")
        for row in range(2):
            axes[row, column].set_ylim(-0.03, 1.03)
            axes[row, column].grid(color="#D9DDDF", linewidth=0.5, alpha=0.55)
            axes[row, column].set_axisbelow(True)
    axes[0, 0].set_ylabel("normalized sandwich position")
    axes[1, 0].set_ylabel("normalized sandwich position")
    top_bar = fig.colorbar(top_image, ax=axes[0, :], fraction=0.018, pad=0.015)
    top_bar.set_label("mean certificate size")
    bottom_bar = fig.colorbar(bottom_image, ax=axes[1, :], fraction=0.018, pad=0.015)
    bottom_bar.set_label("number of linear ANF terms")
    fig.suptitle("Structure inside the lower--upper sandwich", x=0.06, ha="left",
                 fontsize=14, fontweight="bold")
    fig.subplots_adjust(left=0.07, right=0.91, bottom=0.09, top=0.91, wspace=0.16, hspace=0.18)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(FIGURES / f"sandwich_position_structure.{extension}", dpi=220,
                    bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    table = pd.read_csv(ARTIFACTS / "feature_table.csv")
    if not np.array_equal(table["truth_table"].to_numpy(), np.arange(len(table))):
        raise AssertionError("feature table must be in truth-table order")
    boundary = table["khrapchenko_product"].to_numpy(float)
    leaves = table["decision_tree_leaf_count"].to_numpy(float)
    exact_cover = exact_aon_cover_upper()
    constants = []
    positions = {}
    lowers = {}
    uppers = {}
    position_table = table[[
        "truth_table", "mean_certificate_size", "anf_terms_degree_1",
        "khrapchenko_product", "decision_tree_leaf_count",
    ]].copy()

    for language in LANGUAGES:
        target = table[f"target_{language}"].to_numpy(float) + 1
        upper_candidate = exact_cover if language == "AND_OR_NOT" else None
        lower_scale, upper_anchor, upper_slope, lower, upper, position = envelope(
            target, boundary, leaves, upper_candidate)
        lowers[language] = lower
        uppers[language] = upper
        positions[language] = position
        constants.append({
            "language": language,
            "lower_scale": lower_scale,
            "upper_anchor": upper_anchor,
            "upper_slope": upper_slope,
            "exact_cover_contacts": int(np.sum(np.isclose(target, exact_cover)))
            if language == "AND_OR_NOT" else 0,
            "minimum_lower_slack": float(np.min(target - lower)),
            "minimum_upper_slack": float(np.min(upper - target)),
            "median_band_width": float(np.median(upper - lower)),
            "position_q10": float(np.quantile(position, 0.10)),
            "position_median": float(np.median(position)),
            "position_q90": float(np.quantile(position, 0.90)),
            "lower_contacts": int(np.sum(np.isclose(target, lower))),
            "upper_contacts": int(np.sum(np.isclose(target, upper))),
            "collapsed_intervals": int(np.sum(np.isclose(upper, lower))),
        })
        position_table[f"target_{language}"] = target - 1
        position_table[f"lower_{language}"] = lower
        position_table[f"upper_{language}"] = upper
        position_table[f"position_{language}"] = position
    position_table["exact_aon_cover_upper"] = exact_cover

    constants_frame = pd.DataFrame(constants)
    constants_frame.to_csv(ARTIFACTS / "complexity_sandwich_constants.csv", index=False)
    position_table.to_csv(ARTIFACTS / "complexity_sandwich_positions.csv", index=False)
    contacts = contact_frame(table, constants_frame, lowers, uppers, exact_cover)
    contacts.to_csv(ARTIFACTS / "complexity_sandwich_contacts.csv", index=False)

    fold_results, compact_classes = fit_residual_models(table, positions, lowers, uppers)
    fold_results.to_csv(ARTIFACTS / "sandwich_residual_model_folds.csv", index=False)
    model_summary = (fold_results.groupby(["language", "model", "features"], as_index=False)
                     .agg(position_r2_mean=("position_r2", "mean"),
                          position_r2_std=("position_r2", "std"),
                          gate_count_r2_mean=("gate_count_r2", "mean"),
                          gate_count_r2_std=("gate_count_r2", "std")))
    model_summary.to_csv(ARTIFACTS / "sandwich_residual_model_summary.csv", index=False)

    plot_sandwich(table, lowers, uppers)
    plot_sandwich_v2(table, lowers, uppers, exact_cover, constants_frame)
    plot_sandwich_v2(
        table, lowers, uppers, exact_cover, constants_frame,
        stem="semantic_complexity_envelopes",
        title="Semantic complexity envelopes: general bounds and sharp finite constants")
    plot_positions(table, positions)

    best_compact = model_summary[model_summary.model == "compact sandwich"]
    analysis = {
        "functions": len(table),
        "compact_profile_classes": compact_classes,
        "folds": FOLDS,
        "constants": constants,
        "validation": {
            "all_lower_bounds_hold": bool(all(np.all(
                lowers[language] <= table[f"target_{language}"].to_numpy() + 1 + 1e-10)
                for language in LANGUAGES)),
            "all_upper_bounds_hold": bool(all(np.all(
                uppers[language] >= table[f"target_{language}"].to_numpy() + 1 - 1e-10)
                for language in LANGUAGES)),
            "aon_exact_cover_is_constructive": bool(np.all(
                exact_cover >= table["target_AND_OR_NOT"].to_numpy() + 1)),
            "general_decision_tree_bounds_hold": bool(
                np.all(table["target_NAND"].to_numpy() + 1
                       <= 6 + 9 * (leaves - 1))
                and np.all(table["target_NOR"].to_numpy() + 1
                           <= 6 + 9 * (leaves - 1))
                and np.all(table["target_AND_OR_NOT"].to_numpy() + 1
                           <= 3 + 6 * (leaves - 1))),
            "contact_rows": len(contacts),
            "contact_orbits": int(contacts["phase_permutation_orbit_hex"].nunique()),
        },
        "compact_model_gate_count_r2": dict(zip(
            best_compact.language, best_compact.gate_count_r2_mean)),
        "compact_model_position_r2": dict(zip(
            best_compact.language, best_compact.position_r2_mean)),
    }
    (ARTIFACTS / "complexity_sandwich_analysis.json").write_text(
        json.dumps(analysis, indent=2), encoding="utf-8")

    constant_display = constants_frame.copy()
    numeric = constant_display.select_dtypes(include="number").columns
    constant_display[numeric] = constant_display[numeric].round(4)
    model_display = model_summary.copy()
    numeric = model_display.select_dtypes(include="number").columns
    model_display[numeric] = model_display[numeric].round(4)
    contact_summary = (contacts.groupby(["language", "contact"], as_index=False)
                       .agg(functions=("truth_table", "size"),
                            phase_permutation_orbits=(
                                "phase_permutation_orbit_hex", "nunique")))
    report = [
        "# Semantic lower--upper sandwich", "",
        "For every four-input function, this analysis verifies", "",
        "```text",
        "B(f) <= K_L(f) + 1 <= A_L + C_L (U(f) - 1).",
        "```", "",
        "Here `B` is the Khrapchenko product and `U` is minimum deterministic",
        "decision-tree leaf count. The displayed `A` and `C` are the sharp constants",
        "for this affine parameterization on the finite four-input universe; they are",
        "empirical extremal constants, not asymptotic theorems.", "",
        "![Complexity sandwich](../figures/complexity_sandwich.png)", "",
        "![Complexity sandwich v2](../figures/complexity_sandwich_v2.png)", "",
        "## Tight empirical envelopes", "",
        constant_display.to_markdown(index=False), "",
        "Thus NAND and NOR use `B <= K + 1 <= 6 + (9/4)(U - 1)`. The",
        "AND/OR/NOT envelope is", "",
        "```text",
        "B <= K + 1 <= min(3 + (13/9)(U - 1), Q_min).",
        "```", "",
        "`Q_min` is the exact minimum among prime DNF/CNF cover constructions,",
        "including variants that retain one outer NOT. Cube weights account for",
        "literal polarity, internal binary gates, cover-combining gates, and the",
        "outer NOT when present. Constants are assigned cost `K + 1 = 3` because",
        "the language provides no free constants.", "",
        "## Mathematical status", "",
        "The lower inequality `B <= K + 1` holds generally in all three languages.",
        "Push negations to literals to obtain a De Morgan formula without changing",
        "its leaves. Khrapchenko gives `B <= leaves`; a tree with `b` binary gates",
        "has `b + 1` leaves, and `b + 1 <= K + 1` even when unary NOT gates occur.", "",
        "Straight Shannon compilation also gives general, constructive but looser",
        "upper bounds:", "",
        "```text",
        "NAND/NOR:    K + 1 <= 6 + 9(U - 1)",
        "AND/OR/NOT:  K + 1 <= 3 + 6(U - 1).",
        "```", "",
        "A decision-tree leaf is replaced by a constant formula (at most five gates",
        "for NAND/NOR and two for AND/OR/NOT), and each internal Shannon multiplexer",
        "adds four gates. `Q_min` is also constructive for every function. In",
        "contrast, the much sharper slopes `9/4` and `13/9` are fitted extremal",
        "constants on `n = 4`, not asymptotic theorems.", "",
        "## Equality cases", "",
        contact_summary.to_markdown(index=False), "",
        "Every contact function, its phase/permutation orbit, ANF, active upper",
        "construction, and cover witness is recorded in",
        "`../artifacts/complexity_sandwich_contacts.csv`.", "",
        "## Structure within the sandwich", "",
        "![Sandwich position](../figures/sandwich_position_structure.png)", "",
        "The normalized position is `(K + 1 - lower) / (upper - lower)`. Collapsed",
        "exact intervals are assigned the immaterial convention `position = 0.5`.",
        "Color shows",
        "two proposed correction variables: mean certificate size and linear ANF",
        "support.", "",
        "## Held-out residual models", "",
        f"All models use the same five folds over {compact_classes:,} complete compact-profile classes.",
        "The gate-count score reconstructs `K + 1` from the predicted normalized",
        "position and the two envelope curves.", "",
        model_display.to_markdown(index=False), "",
        "The table is generated from the same run as the envelopes, so reported",
        "residual scores cannot silently remain stale after a bound changes. The",
        "compact model combines the two bracket coordinates, certificate summaries,",
        "ANF support, and signed Fourier counts.", "",
        "## Interpretation", "",
        "The experiment tests a concrete decomposition: boundary complexity supplies",
        "a lower obstruction, decision-tree decomposability supplies an upper",
        "construction, and certificates plus algebraic phase locate the exact cost",
        "inside that interval. High held-out residual-model scores support this",
        "decomposition; imperfect scores identify the remaining theoretical gap.", "",
        "## Reproduce", "", "```powershell",
        "python experiments/four_bit/scripts/analyze_complexity_sandwich.py",
        "```", "",
    ]
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "COMPLEXITY_SANDWICH.md").write_text("\n".join(report), encoding="utf-8")

    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
