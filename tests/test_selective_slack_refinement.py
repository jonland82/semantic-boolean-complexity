from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "selective_slack_refinement"
ARTIFACTS = EXPERIMENT / "artifacts"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SelectiveSlackRefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frozen = json.loads(
            (ARTIFACTS / "frozen_cost14_protocol.json").read_text())
        cls.analysis = json.loads(
            (ARTIFACTS / "cost14_prospective_analysis.json").read_text())
        cls.predictions = pd.read_csv(
            ARTIFACTS / "cost14_prospective_predictions.csv")

    def test_frozen_protocol_and_model_hashes_match(self) -> None:
        frozen_path = ARTIFACTS / "frozen_cost14_protocol.json"
        self.assertEqual(
            sha256(frozen_path), self.analysis["frozen_protocol_sha256"])
        self.assertEqual(
            sha256(ARTIFACTS / self.frozen["teacher_artifact"]),
            self.frozen["teacher_sha256"],
        )
        self.assertEqual(
            sha256(ARTIFACTS / self.frozen["student_artifact"]),
            self.frozen["student_sha256"],
        )

    def test_cost_fourteen_sample_obeys_frozen_cohorts(self) -> None:
        config = self.frozen["cost14"]
        self.assertTrue(np.all(
            self.predictions.exact_minimum_gates == config["exact_cost"]))
        counts = self.predictions.cohort.value_counts().to_dict()
        self.assertEqual(counts["targeted"], config["targeted_size"])
        self.assertEqual(counts["random_reference"], config["reference_size"])

    def test_selective_endpoint_only_tightens_construction(self) -> None:
        self.assertTrue(np.all(
            self.predictions.selective_upper_k_plus_1
            <= self.predictions.construction_upper_k_plus_1 + 1e-10))
        self.assertTrue(np.all(self.predictions.selective_reduction >= 0))

    def test_frozen_success_criteria_recompute(self) -> None:
        criteria = self.frozen["cost14"]["success_criteria"]
        covered = (
            self.predictions.exact_k_plus_1
            <= self.predictions.selective_upper_k_plus_1 + 1e-10)
        violation = np.maximum(
            0.0,
            self.predictions.exact_k_plus_1
            - self.predictions.selective_upper_k_plus_1,
        )
        coverage = float(covered.mean())
        maximum_violation = float(violation.max())
        mean_reduction = float(self.predictions.selective_reduction.mean())
        improved_fraction = float(
            (self.predictions.selective_reduction > 0).mean())
        passed = (
            coverage >= criteria["coverage"]
            and maximum_violation <= criteria["maximum_violation"]
            and mean_reduction > 0
            and improved_fraction >= criteria["minimum_improved_fraction"]
        )

        overall = self.analysis["overall"]
        self.assertAlmostEqual(coverage, overall["coverage"])
        self.assertAlmostEqual(
            maximum_violation, overall["maximum_violation"])
        self.assertAlmostEqual(
            mean_reduction, overall["mean_reduction_beyond_construction"])
        self.assertAlmostEqual(
            improved_fraction, overall["improved_fraction"])
        self.assertEqual(
            passed, overall["passed_frozen_success_criteria"])
        self.assertFalse(passed)


if __name__ == "__main__":
    unittest.main()
