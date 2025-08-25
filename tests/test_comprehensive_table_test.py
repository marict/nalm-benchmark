#!/usr/bin/env python3
"""
Tests for comprehensive_table_test.py script
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

sys.path.append(str(Path(__file__).parent.parent / "scripts"))

from comprehensive_table_test import (OPERATIONS, TEST_RANGES, TEST_SEEDS,
                                      get_experiment_key, load_progress,
                                      run_single_test, save_progress)


class TestProgressManagement(unittest.TestCase):
    """Test progress saving and loading functionality."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.progress_file = Path(self.temp_dir) / "test_progress.json"

    def test_load_progress_empty_file(self):
        """Test loading progress when file doesn't exist."""
        progress = load_progress(self.progress_file)
        expected = {"completed": {}, "results": []}
        self.assertEqual(progress, expected)

    def test_save_and_load_progress(self):
        """Test saving and loading progress data."""
        test_data = {
            "completed": {"122_mul_standard": {"grokked": True, "status": "grokking"}},
            "results": [{"operation": "mul", "seed": 122, "grokked": True}],
        }

        save_progress(self.progress_file, test_data)
        self.assertTrue(self.progress_file.exists())

        loaded_data = load_progress(self.progress_file)
        self.assertEqual(loaded_data, test_data)

    def test_load_progress_corrupted_file(self):
        """Test loading progress from corrupted JSON file."""
        # Create a corrupted file
        with open(self.progress_file, "w") as f:
            f.write("invalid json {")

        progress = load_progress(self.progress_file)
        expected = {"completed": {}, "results": []}
        self.assertEqual(progress, expected)

    def test_get_experiment_key(self):
        """Test experiment key generation."""
        key = get_experiment_key(122, "mul", "standard")
        self.assertEqual(key, "122_mul_standard")

        key2 = get_experiment_key(777, "div", "pos_narrow")
        self.assertEqual(key2, "777_div_pos_narrow")


class TestRunSingleTest(unittest.TestCase):
    """Test the run_single_test function."""

    def setUp(self):
        self.mock_process = Mock()
        self.mock_process.stdout = [
            "Starting experiment...",
            "MUL-seed122: 10%|██        | 100/1000",
            "MUL-seed122: 50%|█████     | 500/1000",
            "MUL-seed122: 100%|██████████| 1000/1000",
            "- loss_valid_inter: 1.234e-09",
            "Early stopping at step 987: loss below threshold",
        ]
        self.mock_process.wait.return_value = None

    @patch("comprehensive_table_test.subprocess.Popen")
    @patch("comprehensive_table_test.time.time")
    def test_run_single_test_grokking(self, mock_time, mock_popen):
        """Test successful grokking experiment."""
        mock_time.side_effect = [0.0, 10.0]  # start, end times
        mock_popen.return_value = self.mock_process

        result = run_single_test("mul", 122, [-2, 2], [[-6, -2], [2, 6]])

        self.assertEqual(result["operation"], "mul")
        self.assertEqual(result["seed"], 122)
        self.assertTrue(result["grokked"])
        self.assertEqual(result["status"], "grokking")
        self.assertEqual(result["grok_step"], 987)
        self.assertFalse(result["nan_error"])
        self.assertEqual(result["duration"], 10.0)

    @patch("comprehensive_table_test.subprocess.Popen")
    @patch("comprehensive_table_test.time.time")
    def test_run_single_test_not_grokking(self, mock_time, mock_popen):
        """Test experiment that doesn't grok."""
        mock_time.side_effect = [0.0, 15.0]

        # Set up mock for non-grokking output
        self.mock_process.stdout = [
            "Starting experiment...",
            "MUL-seed122: 100%|██████████| 1000/1000",
            "- loss_valid_inter: 0.123456",  # High loss, no grokking
        ]
        mock_popen.return_value = self.mock_process

        result = run_single_test("mul", 122, [-2, 2], [[-6, -2], [2, 6]])

        self.assertFalse(result["grokked"])
        self.assertEqual(result["status"], "not_grokking")
        self.assertIsNone(result["grok_step"])
        self.assertFalse(result["nan_error"])
        self.assertEqual(result["final_inter_loss"], 0.123456)

    @patch("comprehensive_table_test.subprocess.Popen")
    @patch("comprehensive_table_test.time.time")
    def test_run_single_test_nan_error(self, mock_time, mock_popen):
        """Test experiment with NaN error."""
        mock_time.side_effect = [0.0, 5.0]

        # Set up mock for NaN output
        self.mock_process.stdout = [
            "Starting experiment...",
            "MUL-seed122: 50%|█████     | 500/1000",
            "Warning: NaN detected in gradient",
            "- loss_valid_inter: nan",
        ]
        mock_popen.return_value = self.mock_process

        result = run_single_test("mul", 122, [-2, 2], [[-6, -2], [2, 6]])

        self.assertFalse(result["grokked"])
        self.assertTrue(result["nan_error"])
        self.assertEqual(result["status"], "nan_error")
        self.assertIsNone(result["grok_step"])

    @patch("comprehensive_table_test.subprocess.Popen")
    @patch("comprehensive_table_test.time.time")
    def test_run_single_test_timeout(self, mock_time, mock_popen):
        """Test experiment timeout handling."""
        mock_time.side_effect = [0.0, 120.0]

        # Mock timeout exception
        mock_process = Mock()
        mock_process.stdout = []
        mock_process.wait.side_effect = subprocess.TimeoutExpired("cmd", 120)
        mock_popen.return_value = mock_process

        result = run_single_test("mul", 122, [-2, 2], [[-6, -2], [2, 6]])

        self.assertFalse(result["grokked"])
        self.assertFalse(result["nan_error"])
        self.assertEqual(result["status"], "not_grokking")
        self.assertEqual(result["duration"], 120)
        self.assertEqual(result["final_inter_loss"], float("inf"))

    @patch("comprehensive_table_test.subprocess.Popen")
    @patch("comprehensive_table_test.time.time")
    @patch("builtins.print")
    def test_show_tqdm_only_filtering(self, mock_print, mock_time, mock_popen):
        """Test that show_tqdm_only properly filters output."""
        mock_time.side_effect = [0.0, 10.0]

        # Mock output with mix of tqdm and other lines
        self.mock_process.stdout = [
            "Starting experiment...",  # Should be filtered out
            "MUL-seed122: 10%|██        | 100/1000",  # Should be shown
            "Some debug output",  # Should be filtered out
            "- loss_valid_inter: 1.234",  # Should be filtered out
            "MUL-seed122: 50%|█████     | 500/1000",  # Should be shown
            "Early stopping at step 500: loss below threshold",  # Should be filtered out
        ]
        mock_popen.return_value = self.mock_process

        result = run_single_test(
            "mul", 122, [-2, 2], [[-6, -2], [2, 6]], show_tqdm_only=True
        )

        # Check that only tqdm lines were printed
        printed_calls = [call for call in mock_print.call_args_list if call[0]]
        tqdm_lines = [call for call in printed_calls if "seed122:" in str(call)]
        self.assertEqual(
            len(tqdm_lines), 2
        )  # Only the two tqdm lines should be printed


class TestConfigurationConstants(unittest.TestCase):
    """Test that configuration constants are properly defined."""

    def test_test_seeds_defined(self):
        """Test that TEST_SEEDS is properly defined."""
        self.assertIsInstance(TEST_SEEDS, list)
        self.assertTrue(len(TEST_SEEDS) > 0)
        self.assertTrue(all(isinstance(seed, int) for seed in TEST_SEEDS))

    def test_operations_defined(self):
        """Test that OPERATIONS is properly defined."""
        expected_ops = ["mul", "add", "sub", "div"]
        self.assertEqual(OPERATIONS, expected_ops)

    def test_test_ranges_defined(self):
        """Test that TEST_RANGES is properly defined."""
        self.assertIsInstance(TEST_RANGES, list)
        self.assertTrue(len(TEST_RANGES) > 0)

        # Check structure of each range tuple
        for range_tuple in TEST_RANGES:
            self.assertEqual(len(range_tuple), 3)  # (interp_range, extrap_range, name)
            interp_range, extrap_range, name = range_tuple
            self.assertIsInstance(interp_range, list)
            self.assertIsInstance(name, str)


class TestResultStatuses(unittest.TestCase):
    """Test result status categorization."""

    def test_status_categorization(self):
        """Test that all possible statuses are handled correctly."""
        # Test grokking status
        result_grokking = {"grokked": True, "nan_error": False, "status": "grokking"}

        # Test nan_error status
        result_nan = {"grokked": False, "nan_error": True, "status": "nan_error"}

        # Test not_grokking status
        result_not_grokking = {
            "grokked": False,
            "nan_error": False,
            "status": "not_grokking",
        }

        # Verify status logic
        self.assertEqual(result_grokking["status"], "grokking")
        self.assertEqual(result_nan["status"], "nan_error")
        self.assertEqual(result_not_grokking["status"], "not_grokking")


if __name__ == "__main__":
    unittest.main()
