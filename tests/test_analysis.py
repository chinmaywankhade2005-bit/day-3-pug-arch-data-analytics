"""Edge-case and workflow tests for the Day 3 facility analysis."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import analysis


class CleaningTests(unittest.TestCase):
    def test_sample_cleans_known_issues_and_preserves_raw(self) -> None:
        raw = pd.read_csv(analysis.RAW_PATH)
        original = raw.copy(deep=True)
        cleaned, log, details, issues = analysis.clean_data(raw)
        self.assertEqual(raw.shape, original.shape)
        pd.testing.assert_frame_equal(raw, original)
        self.assertEqual(details["input_rows"], 50)
        self.assertEqual(details["exact_duplicate_rows_removed"], 1)
        self.assertEqual(details["rows_dropped_for_missing_id_or_invalid_date"], 1)
        self.assertEqual(details["output_rows"], 48)
        self.assertEqual(details["invalid_score_values"], 1)
        self.assertEqual(details["invalid_negative_count_values"], 1)
        self.assertGreater(details["rows_with_at_least_one_outlier"], 0)
        self.assertEqual(int(cleaned.isna().sum().sum()), 0)
        self.assertIn("numeric_outlier_rows", set(log["issue"]))
        self.assertIn("iqr_outlier", set(issues["issue"]))
        self.assertIn("exact_duplicate", set(issues["issue"]))
        self.assertTrue(cleaned["cleanliness_score"].between(0, 100).all())

    def test_empty_dataset_is_rejected(self) -> None:
        empty = pd.DataFrame(columns=analysis.EXPECTED_COLUMNS)
        with self.assertRaisesRegex(ValueError, "empty"):
            analysis.clean_data(empty)

    def test_missing_required_column_is_rejected(self) -> None:
        incomplete = pd.DataFrame(columns=analysis.EXPECTED_COLUMNS[:-1])
        with self.assertRaisesRegex(ValueError, "Missing required columns"):
            analysis.clean_data(incomplete)

    def test_unrepairable_rows_are_reported_when_no_valid_dates_remain(self) -> None:
        row = {column: "" for column in analysis.EXPECTED_COLUMNS}
        row.update({"facility_id": "FAC-X", "location": "Test", "cleanliness_score": 80,
                    "odor_score": 10, "waste_level": 10, "water_availability": "Available",
                    "footfall": 3, "complaints": 0, "inspection_date": "invalid-date"})
        with self.assertRaisesRegex(ValueError, "No valid records remain"):
            analysis.clean_data(pd.DataFrame([row]))

    def test_full_run_creates_expected_artifacts_without_touching_source(self) -> None:
        with tempfile.TemporaryDirectory(dir=analysis.ROOT) as temp_dir:
            raw_path = Path(temp_dir) / "input.csv"
            output_dir = Path(temp_dir) / "outputs"
            pd.read_csv(analysis.RAW_PATH).to_csv(raw_path, index=False)
            before = raw_path.read_bytes()
            result = analysis.run_analysis(raw_path, output_dir)
            self.assertEqual(raw_path.read_bytes(), before)
            self.assertEqual(len(result["charts"]), 5)
            self.assertEqual(len(result["summary"]["insights"]), 3)
            self.assertTrue((output_dir / "quality_issues.csv").exists())
            summary = json.loads((output_dir / "analysis_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["cleaned_record_count"], 48)
            self.assertTrue(all(Path(chart).stat().st_size > 0 for chart in result["charts"]))
            with self.assertRaises(FileExistsError):
                analysis.run_analysis(raw_path, output_dir)


if __name__ == "__main__":
    unittest.main()
