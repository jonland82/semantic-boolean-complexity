"""Compare the empirical sandwich constants on three and four inputs."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENT.parents[1]
ARTIFACTS = EXPERIMENT / "artifacts"


def load_module(name: str, filename: str):
    path = EXPERIMENT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def khrapchenko_products(n: int) -> np.ndarray:
    assignments = np.arange(1 << n)
    tables = np.arange(1 << (1 << n), dtype=np.uint32)
    bits = ((tables[:, None] >> assignments) & 1).astype(np.int8)
    local = np.zeros_like(bits)
    for variable in range(n):
        local += bits != bits[:, assignments ^ (1 << variable)]
    zeros = np.sum(bits == 0, axis=1)
    ones = np.sum(bits == 1, axis=1)
    average_zero = np.divide(
        np.sum(local * (bits == 0), axis=1), zeros,
        out=np.zeros(len(bits), dtype=float), where=zeros != 0)
    average_one = np.divide(
        np.sum(local * (bits == 1), axis=1), ones,
        out=np.zeros(len(bits), dtype=float), where=ones != 0)
    return average_zero * average_one


def main() -> None:
    sandwich = load_module("sandwich", "analyze_complexity_sandwich.py")
    associations = load_module("associations", "analyze_feature_associations.py")
    reference = pd.read_csv(ROOT / "data" / "reference" / "three_bit_minima.csv")
    boundary = khrapchenko_products(3)
    leaves = associations.decision_tree_measures(3)[1].astype(float)
    cover = sandwich.exact_aon_cover_upper(3)
    rows = []
    for language in sandwich.LANGUAGES:
        target = (reference[reference.language == language]
                  .sort_values("truth_table").minimum_gates.to_numpy(float) + 1)
        result = sandwich.envelope(
            target, boundary, leaves, cover if language == "AND_OR_NOT" else None)
        scale, anchor, slope, lower, upper, _ = result
        rows.append({
            "inputs": 3,
            "functions": len(target),
            "language": language,
            "lower_scale": scale,
            "upper_anchor": anchor,
            "upper_slope": slope,
            "exact_cover_contacts": int(np.sum(target == cover))
            if language == "AND_OR_NOT" else 0,
            "lower_contacts": int(np.sum(np.isclose(target, lower))),
            "upper_contacts": int(np.sum(np.isclose(target, upper))),
            "median_band_width": float(np.median(upper - lower)),
        })

    four = pd.read_csv(ARTIFACTS / "complexity_sandwich_constants.csv")
    for item in four.to_dict(orient="records"):
        rows.append({
            "inputs": 4,
            "functions": 65536,
            "language": item["language"],
            "lower_scale": item["lower_scale"],
            "upper_anchor": item["upper_anchor"],
            "upper_slope": item["upper_slope"],
            "exact_cover_contacts": item["exact_cover_contacts"],
            "lower_contacts": item["lower_contacts"],
            "upper_contacts": item["upper_contacts"],
            "median_band_width": item["median_band_width"],
        })
    result = pd.DataFrame(rows).sort_values(["inputs", "language"])
    result.to_csv(ARTIFACTS / "sandwich_dimension_comparison.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
