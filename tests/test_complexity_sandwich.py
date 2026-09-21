from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "four_bit"
SCRIPT = EXPERIMENT / "scripts" / "analyze_complexity_sandwich.py"
SPEC = importlib.util.spec_from_file_location("complexity_sandwich", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SANDWICH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SANDWICH)


def truth_table(predicate) -> int:
    return sum(int(bool(predicate(assignment))) << assignment for assignment in range(16))


class ExactCoverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cover = SANDWICH.exact_aon_cover_upper()
        cls.features = pd.read_csv(EXPERIMENT / "artifacts" / "feature_table.csv")

    def test_elementary_formula_costs(self) -> None:
        x0 = truth_table(lambda assignment: assignment & 1)
        not_x0 = truth_table(lambda assignment: not (assignment & 1))
        conjunction = truth_table(lambda assignment: (assignment & 1) and (assignment & 2))
        disjunction = truth_table(lambda assignment: (assignment & 1) or (assignment & 2))
        self.assertEqual(int(self.cover[0]), 3)
        self.assertEqual(int(self.cover[-1]), 3)
        self.assertEqual(int(self.cover[x0]), 1)
        self.assertEqual(int(self.cover[not_x0]), 2)
        self.assertEqual(int(self.cover[conjunction]), 2)
        self.assertEqual(int(self.cover[disjunction]), 2)

    def test_cover_is_an_exhaustive_constructive_upper_bound(self) -> None:
        exact = self.features["target_AND_OR_NOT"].to_numpy(int) + 1
        self.assertTrue(np.all(self.cover >= exact))
        self.assertEqual(int(np.sum(self.cover == exact)), 986)


class GeneratedArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.features = pd.read_csv(EXPERIMENT / "artifacts" / "feature_table.csv")
        cls.positions = pd.read_csv(
            EXPERIMENT / "artifacts" / "complexity_sandwich_positions.csv")
        cls.constants = pd.read_csv(
            EXPERIMENT / "artifacts" / "complexity_sandwich_constants.csv")
        cls.contacts = pd.read_csv(
            EXPERIMENT / "artifacts" / "complexity_sandwich_contacts.csv")
        cls.analysis = json.loads((
            EXPERIMENT / "artifacts" / "complexity_sandwich_analysis.json"
        ).read_text(encoding="utf-8"))

    def test_every_generated_interval_contains_exact_complexity(self) -> None:
        for language in SANDWICH.LANGUAGES:
            exact = self.features[f"target_{language}"].to_numpy(float) + 1
            lower = self.positions[f"lower_{language}"].to_numpy(float)
            upper = self.positions[f"upper_{language}"].to_numpy(float)
            self.assertTrue(np.all(lower <= exact + 1e-10), language)
            self.assertTrue(np.all(upper >= exact - 1e-10), language)
            self.assertFalse(self.positions[f"position_{language}"].isna().any())

    def test_general_constructive_decision_tree_bounds(self) -> None:
        leaves = self.features.decision_tree_leaf_count.to_numpy(int)
        for language, anchor, slope in (
                ("NAND", 6, 9), ("NOR", 6, 9), ("AND_OR_NOT", 3, 6)):
            exact = self.features[f"target_{language}"].to_numpy(int) + 1
            self.assertTrue(np.all(exact <= anchor + slope * (leaves - 1)))

    def test_json_and_csv_constants_are_synchronized(self) -> None:
        json_constants = {row["language"]: row
                          for row in self.analysis["constants"]}
        for row in self.constants.to_dict(orient="records"):
            expected = json_constants[row["language"]]
            self.assertEqual(set(row), set(expected))
            for key, value in row.items():
                if key == "language":
                    continue
                self.assertAlmostEqual(float(value), float(expected[key]), places=12)
        for key in ("all_lower_bounds_hold", "all_upper_bounds_hold",
                    "aon_exact_cover_is_constructive",
                    "general_decision_tree_bounds_hold"):
            self.assertTrue(self.analysis["validation"][key], key)

    def test_contact_counts_are_synchronized(self) -> None:
        for row in self.constants.itertuples(index=False):
            contacts = self.contacts[self.contacts.language == row.language]
            lower = contacts.contact.isin(["lower", "both"]).sum()
            upper = contacts.contact.isin(["upper", "both"]).sum()
            self.assertEqual(lower, row.lower_contacts)
            self.assertEqual(upper, row.upper_contacts)

    def test_cover_contact_witnesses_match_generated_upper(self) -> None:
        cover_contacts = self.contacts[
            self.contacts.active_upper_source.str.contains("exact prime cover")]
        self.assertFalse(cover_contacts.empty)
        self.assertTrue(np.allclose(
            cover_contacts.cover_cost_k_plus_1.to_numpy(float),
            self.positions.loc[
                cover_contacts.truth_table, "exact_aon_cover_upper"
            ].to_numpy(float),
        ))

    def test_three_input_dimensional_check(self) -> None:
        comparison = pd.read_csv(
            EXPERIMENT / "artifacts" / "sandwich_dimension_comparison.csv")
        three = comparison[comparison.inputs == 3].set_index("language")
        self.assertEqual(len(three), 3)
        self.assertAlmostEqual(three.loc["NAND", "upper_slope"], 5 / 3)
        self.assertAlmostEqual(three.loc["NOR", "upper_slope"], 5 / 3)
        self.assertAlmostEqual(three.loc["AND_OR_NOT", "upper_slope"], 10 / 7)
        self.assertTrue(np.allclose(three.lower_scale, 1))


if __name__ == "__main__":
    unittest.main()
