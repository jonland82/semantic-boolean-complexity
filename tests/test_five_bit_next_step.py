from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "five_bit_next_step"
SCRIPT = EXPERIMENT / "scripts" / "run_aon_upper_tail.py"
SPEC = importlib.util.spec_from_file_location("five_bit_next_step", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
NEXT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(NEXT)


class FiveBitNextStepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.selected = pd.read_csv(
            EXPERIMENT / "artifacts" / "aon_cost_12_selected.csv")

    def test_protocol_hash_matches_generated_metadata(self) -> None:
        protocol_bytes = (EXPERIMENT / "protocol.json").read_bytes()
        metadata = json.loads(
            (EXPERIMENT / "artifacts" / "aon_cost_12_analysis.json").read_text())
        self.assertEqual(
            hashlib.sha256(protocol_bytes).hexdigest(),
            metadata["protocol_sha256"],
        )

    def test_selected_targets_and_cohorts_follow_protocol(self) -> None:
        protocol = json.loads((EXPERIMENT / "protocol.json").read_text())
        self.assertTrue(np.all(
            self.selected.exact_minimum_gates == protocol["test_cost"]))
        counts = self.selected.cohort.value_counts().to_dict()
        self.assertEqual(counts["upper_tail"], protocol["upper_tail_size"])
        self.assertEqual(counts["random_reference"], protocol["reference_size"])

    def test_certified_constructions_are_valid_upper_bounds(self) -> None:
        exact_k_plus_1 = self.selected.exact_minimum_gates + 1
        self.assertTrue(np.all(
            self.selected.construction_upper_k_plus_1 >= exact_k_plus_1))
        self.assertTrue(np.all(
            self.selected.construction_upper_k_plus_1
            <= self.selected.classical_upper_k_plus_1))

    def test_learned_intervals_stay_inside_analytic_envelope(self) -> None:
        for method in ("baseline", "structure_augmented"):
            self.assertTrue(np.all(
                self.selected[f"{method}_lower_k_plus_1"]
                >= self.selected.classical_lower_k_plus_1 - 1e-10))
            self.assertTrue(np.all(
                self.selected[f"{method}_upper_k_plus_1"]
                <= self.selected.classical_upper_k_plus_1 + 1e-10))

    def test_factor_bound_detects_simple_disjoint_and(self) -> None:
        four = pd.read_csv(
            ROOT / "experiments" / "four_bit" / "artifacts" / "feature_table.csv")
        costs = four.target_AND_OR_NOT.to_numpy(int)
        variables = [sum(((assignment >> variable) & 1) << assignment
                         for assignment in range(32))
                     for variable in range(5)]
        features = NEXT.construction_features(variables[0] & variables[1], 5, costs)
        self.assertEqual(features["factor_upper_k_plus_1"], 2.0)


if __name__ == "__main__":
    unittest.main()
