from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "five_bit_pilot" / "scripts" / "run_local_pilot.py"
SPEC = importlib.util.spec_from_file_location("five_bit_pilot", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PILOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PILOT)


class FiveBitPilotTests(unittest.TestCase):
    def test_generic_features_reproduce_four_bit_artifact(self) -> None:
        reference = pd.read_csv(
            ROOT / "experiments" / "four_bit" / "artifacts" / "feature_table.csv",
            index_col="truth_table")
        core = PILOT.load_core()
        for table in (0, 1, 0x6996, 0xA5C3, 0xFFFF):
            observed = PILOT.compact_features(table, 4)
            expected = reference.loc[table]
            for feature in core.COMPACT:
                self.assertAlmostEqual(
                    observed[feature], float(expected[feature]), places=10,
                    msg=f"table={table:#06x}, feature={feature}")

    def test_sparse_layers_are_exact_and_disjoint_through_three(self) -> None:
        for language in PILOT.LANGUAGES:
            layers, _ = PILOT.exact_sparse_layers(language, 5, 3)
            flattened = [value for layer in layers for value in layer]
            self.assertEqual(len(flattened), len(set(flattened)))
            self.assertTrue(all(len(layer) > 0 for layer in layers))

    def test_generated_pilot_targets_respect_exact_layer_limit(self) -> None:
        predictions = pd.read_csv(
            ROOT / "experiments" / "five_bit_pilot" / "artifacts"
            / "five_bit_pilot_predictions.csv")
        self.assertTrue(np.all(predictions.exact_minimum_gates <= PILOT.MAX_COST))
        self.assertTrue(np.all(predictions.exact_minimum_gates >= 0))

    def test_dimension_calibration_keeps_cost_ten_as_holdout(self) -> None:
        holdout = pd.read_csv(
            ROOT / "experiments" / "five_bit_pilot" / "artifacts"
            / "five_bit_cost_10_holdout.csv")
        summary = pd.read_csv(
            ROOT / "experiments" / "five_bit_pilot" / "artifacts"
            / "five_bit_dimension_calibration_summary.csv")
        self.assertTrue(np.all(holdout.exact_minimum_gates == 10))
        self.assertEqual(len(summary), len(PILOT.LANGUAGES) * 3)
        self.assertEqual(set(summary.method), {
            "n4_zero_shot", "n5_cost_0_8_calibrated",
            "n5_affine_khat_adaptation",
        })

    def test_conditional_calibration_keeps_cost_eleven_prospective(self) -> None:
        holdout = pd.read_csv(
            ROOT / "experiments" / "five_bit_pilot" / "artifacts"
            / "five_bit_cost_11_prospective.csv")
        summary = pd.read_csv(
            ROOT / "experiments" / "five_bit_pilot" / "artifacts"
            / "five_bit_conditional_calibration_summary.csv")
        self.assertTrue(np.all(holdout.exact_minimum_gates == 11))
        self.assertEqual(len(summary), len(PILOT.LANGUAGES))
        self.assertTrue(np.all(
            holdout.conditional_lower_k_plus_1
            >= holdout.classical_lower_k_plus_1 - 1e-10))
        self.assertTrue(np.all(
            holdout.conditional_upper_k_plus_1
            <= holdout.classical_upper_k_plus_1 + 1e-10))


if __name__ == "__main__":
    unittest.main()
