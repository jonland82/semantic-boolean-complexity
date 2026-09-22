from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "residual_slack_refinement"
ARTIFACTS = EXPERIMENT / "artifacts"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ResidualSlackRefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frozen = json.loads(
            (ARTIFACTS / "frozen_cost13_protocol.json").read_text())
        cls.analysis = json.loads(
            (ARTIFACTS / "cost13_prospective_analysis.json").read_text())
        cls.predictions = pd.read_csv(
            ARTIFACTS / "cost13_prospective_predictions.csv")

    def test_frozen_protocol_and_model_hashes_match(self) -> None:
        frozen_path = ARTIFACTS / "frozen_cost13_protocol.json"
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

    def test_cost_thirteen_sample_obeys_frozen_cohorts(self) -> None:
        config = self.frozen["cost13"]
        self.assertTrue(np.all(
            self.predictions.exact_minimum_gates == config["exact_cost"]))
        counts = self.predictions.cohort.value_counts().to_dict()
        self.assertEqual(counts["targeted"], config["targeted_size"])
        self.assertEqual(counts["random_reference"], config["reference_size"])

    def test_refined_endpoint_only_tightens_construction(self) -> None:
        self.assertTrue(np.all(
            self.predictions.refined_upper_k_plus_1
            <= self.predictions.construction_upper_k_plus_1 + 1e-10))
        self.assertTrue(np.all(
            self.predictions.learned_reduction_beyond_construction >= 0))

    def test_frozen_success_criteria_recompute(self) -> None:
        criteria = self.frozen["cost13"]["success_criteria"]
        coverage = float(self.predictions.covered.mean())
        maximum_violation = float(self.predictions.violation.max())
        mean_reduction = float(
            self.predictions.learned_reduction_beyond_construction.mean())
        self.assertGreaterEqual(coverage, criteria["coverage"])
        self.assertLessEqual(maximum_violation, criteria["maximum_violation"])
        self.assertGreater(mean_reduction, 0)
        self.assertTrue(
            self.analysis["overall"]["passed_frozen_success_criteria"])


if __name__ == "__main__":
    unittest.main()
