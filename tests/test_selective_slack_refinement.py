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
        cls.casebook = pd.read_csv(
            ARTIFACTS / "cost14_one_slack_casebook.csv")
        cls.support_shift = pd.read_csv(
            ARTIFACTS / "cost14_failure_support_shift.csv")
        cls.local_selection = pd.read_csv(
            ARTIFACTS / "local_selection_aware_summary.csv").set_index("policy")
        cls.local_analysis = json.loads(
            (ARTIFACTS / "local_selection_aware_analysis.json").read_text())

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

    def test_failure_casebook_is_the_matched_one_slack_regime(self) -> None:
        self.assertEqual(len(self.casebook), 10)
        self.assertTrue(np.all(self.casebook.construction_residual == 1))
        self.assertEqual(int((~self.casebook.covered).sum()), 6)
        self.assertEqual(self.casebook.npn_canonical.nunique(), 10)
        ordered = self.casebook.sort_values(
            "predicted_residual_slack", ascending=False)
        self.assertTrue(np.all(~ordered.covered.iloc[:6]))
        self.assertTrue(np.all(ordered.covered.iloc[6:]))

    def test_failure_audit_records_calibration_support_shift(self) -> None:
        support = self.support_shift.set_index("cost")
        self.assertEqual(int(support.loc[13, "one_slack_functions"]), 1)
        self.assertEqual(int(support.loc[13, "functions"]), 46)
        self.assertEqual(int(support.loc[14, "one_slack_functions"]), 9)
        self.assertEqual(int(support.loc[14, "functions"]), 78)
        self.assertGreater(
            support.loc[14, "maximum_dangerous_residual"],
            support.loc[13, "maximum_dangerous_residual"],
        )

    def test_local_boundary_guards_remove_diagnostic_misses(self) -> None:
        baseline = self.local_selection.loc["frozen_baseline"]
        unit_cap = self.local_selection.loc["frozen_unit_reduction_cap"]
        endpoint_guard = self.local_selection.loc["frozen_abstain_endpoint_16"]
        self.assertEqual(int(baseline.misses), 6)
        self.assertEqual(int(unit_cap.misses), 0)
        self.assertEqual(int(endpoint_guard.misses), 0)
        self.assertGreater(unit_cap.fraction_of_baseline_gain, 0.85)

    def test_local_learned_residual_variants_do_not_claim_a_repair(self) -> None:
        learned = self.local_selection.loc[
            self.local_selection.index.str.contains(
                "score_only|construction_aware|full_restriction_witness")
        ]
        self.assertTrue(np.all(learned.misses > 0))
        self.assertEqual(
            self.local_analysis["status"],
            "retrospective diagnostic; not prospective evidence",
        )
        batch = self.local_analysis["max_residual_batch_calibration"]
        self.assertEqual(batch["selected_calibration_cases"], 80)
        self.assertEqual(
            batch["selected_calibration_cases_needed_for_99_percent_batch_probability"],
            7920,
        )


if __name__ == "__main__":
    unittest.main()
