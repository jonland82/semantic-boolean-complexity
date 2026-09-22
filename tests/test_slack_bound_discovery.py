from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "slack_bound_discovery"
ARTIFACTS = EXPERIMENT / "artifacts"


class SlackBoundDiscoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.predictions = pd.read_csv(
            ARTIFACTS / "five_bit_slack_predictions.csv")
        cls.audit = pd.read_csv(ARTIFACTS / "four_bit_finite_audit.csv")

    def test_protocol_hash_matches_analysis(self) -> None:
        protocol_bytes = (EXPERIMENT / "protocol.json").read_bytes()
        analysis = json.loads(
            (ARTIFACTS / "slack_discovery_analysis.json").read_text())
        self.assertEqual(
            hashlib.sha256(protocol_bytes).hexdigest(),
            analysis["protocol_sha256"],
        )

    def test_only_cost_eleven_and_twelve_are_evaluated(self) -> None:
        self.assertEqual(
            set(self.predictions.exact_minimum_gates.astype(int)), {11, 12})

    def test_learned_and_combined_endpoints_only_tighten(self) -> None:
        self.assertTrue(np.all(
            self.predictions.learned_upper_k_plus_1
            <= self.predictions.existing_upper_k_plus_1 + 1e-10))
        self.assertTrue(np.all(
            self.predictions.combined_upper_k_plus_1
            <= self.predictions.learned_upper_k_plus_1 + 1e-10))
        self.assertTrue(np.all(
            self.predictions.combined_upper_k_plus_1
            <= self.predictions.construction_upper_k_plus_1 + 1e-10))

    def test_four_input_combined_audit_is_complete(self) -> None:
        self.assertTrue(np.all(self.audit.functions == 65536))
        self.assertTrue(np.allclose(self.audit.coverage, 1.0))
        self.assertTrue(np.allclose(self.audit.combined_coverage, 1.0))

    def test_boosting_cost_twelve_coverage_is_complete(self) -> None:
        selected = self.predictions[
            (self.predictions.model == "boosting")
            & (self.predictions.exact_minimum_gates == 12)]
        self.assertEqual(len(selected), 120)
        self.assertTrue(np.all(selected.covered))


if __name__ == "__main__":
    unittest.main()
