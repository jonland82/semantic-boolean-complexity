from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "four_bit"
SCRIPT = EXPERIMENT / "scripts" / "analyze_learned_envelope.py"
SPEC = importlib.util.spec_from_file_location("learned_envelope", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
LEARNED = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LEARNED)


class LearnedEnvelopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.predictions = pd.read_csv(
            EXPERIMENT / "artifacts" / "learned_envelope_predictions.csv")
        cls.summary = pd.read_csv(
            EXPERIMENT / "artifacts" / "learned_envelope_summary.csv")

    def test_every_point_prediction_stays_in_classical_envelope(self) -> None:
        self.assertTrue(np.all(
            self.predictions.predicted_k_plus_1
            >= self.predictions.classical_lower_k_plus_1 - 1e-10))
        self.assertTrue(np.all(
            self.predictions.predicted_k_plus_1
            <= self.predictions.classical_upper_k_plus_1 + 1e-10))

    def test_exhaustive_residual_intervals_cover_every_function(self) -> None:
        self.assertTrue(np.all(
            self.predictions.exhaustive_lower_k_plus_1
            <= self.predictions.exact_k_plus_1 + 1e-10))
        self.assertTrue(np.all(
            self.predictions.exhaustive_upper_k_plus_1
            >= self.predictions.exact_k_plus_1 - 1e-10))
        self.assertTrue(np.all(
            self.predictions.exhaustive_integer_lower_k_plus_1
            <= self.predictions.exact_k_plus_1))
        self.assertTrue(np.all(
            self.predictions.exhaustive_integer_upper_k_plus_1
            >= self.predictions.exact_k_plus_1))

    def test_all_learned_intervals_are_subsets_of_classical_envelope(self) -> None:
        for coverage in LEARNED.COVERAGES:
            suffix = str(int(round(100 * coverage)))
            lower = self.predictions[f"calibrated_{suffix}_lower_k_plus_1"]
            upper = self.predictions[f"calibrated_{suffix}_upper_k_plus_1"]
            self.assertTrue(np.all(
                lower >= self.predictions.classical_lower_k_plus_1 - 1e-10))
            self.assertTrue(np.all(
                upper <= self.predictions.classical_upper_k_plus_1 + 1e-10))
            self.assertTrue(np.all(lower <= upper + 1e-10))

    def test_summary_is_complete_and_synchronized(self) -> None:
        self.assertEqual(
            len(self.summary),
            len(LEARNED.LANGUAGES) * (1 + 2 * len(LEARNED.COVERAGES)))
        exhaustive = self.summary[
            self.summary.method == "cross_fitted_exhaustive_max_residual"]
        self.assertTrue(np.allclose(exhaustive.function_coverage, 1.0))
        self.assertTrue(np.allclose(exhaustive.class_coverage, 1.0))

    def test_gap_normalized_intervals_obey_shrinkage_theorem(self) -> None:
        normalized = self.summary[
            self.summary.method == "split_conformal_gap_normalized"]
        self.assertTrue(np.all(
            normalized.mean_relative_shrinkage + 1e-10
            >= normalized.minimum_shrinkage_guarantee))
        self.assertTrue(np.all(
            normalized.minimum_shrinkage_guarantee >= -1e-10))
        lower = self.predictions.gap_normalized_95_lower_k_plus_1
        upper = self.predictions.gap_normalized_95_upper_k_plus_1
        self.assertTrue(np.all(
            lower >= self.predictions.classical_lower_k_plus_1 - 1e-10))
        self.assertTrue(np.all(
            upper <= self.predictions.classical_upper_k_plus_1 + 1e-10))
        for language in LEARNED.LANGUAGES:
            rows = self.predictions[self.predictions.language == language]
            widths = (rows.classical_upper_k_plus_1
                      - rows.classical_lower_k_plus_1)
            learned_widths = (rows.gap_normalized_95_upper_k_plus_1
                              - rows.gap_normalized_95_lower_k_plus_1)
            positive = widths > 1e-10
            pointwise_shrinkage = 1 - learned_widths[positive] / widths[positive]
            guarantee = normalized[
                (normalized.language == language)
                & np.isclose(normalized.nominal_coverage, 0.95)
            ].minimum_shrinkage_guarantee.iloc[0]
            self.assertTrue(np.all(pointwise_shrinkage + 1e-10 >= guarantee))

    def test_conformal_quantile_uses_finite_sample_correction(self) -> None:
        scores = np.arange(1, 101, dtype=float)
        self.assertEqual(LEARNED.conformal_higher_quantile(scores, 0.95), 96.0)


if __name__ == "__main__":
    unittest.main()
