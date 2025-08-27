#!/usr/bin/env python3
"""
Tests for --no-selector functionality in DAG layer
"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.append(str(Path(__file__).parent.parent))

import torch

from stable_nalu.layer.dag import DAGLayer


class TestNoSelectorFunctionality(unittest.TestCase):
    """Test the --no-selector argument and DAG layer behavior."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.in_features = 2
        self.out_features = 1
        self.dag_depth = 3

    def test_dag_layer_no_selector_init(self):
        """Test DAG layer initializes correctly with no_selector=True."""
        layer = DAGLayer(
            in_features=self.in_features,
            out_features=self.out_features,
            dag_depth=self.dag_depth,
            no_selector=True,
        )

        self.assertTrue(layer.no_selector)
        # Should still have output_selector_head for compatibility
        self.assertIsNotNone(layer.output_selector_head)

    def test_dag_layer_no_selector_false(self):
        """Test DAG layer initializes correctly with no_selector=False (default)."""
        layer = DAGLayer(
            in_features=self.in_features,
            out_features=self.out_features,
            dag_depth=self.dag_depth,
            no_selector=False,
        )

        self.assertFalse(layer.no_selector)

    def test_dag_layer_no_selector_default(self):
        """Test DAG layer defaults to no_selector=False."""
        layer = DAGLayer(
            in_features=self.in_features,
            out_features=self.out_features,
            dag_depth=self.dag_depth,
        )

        self.assertFalse(layer.no_selector)

    def test_dag_layer_forward_with_no_selector(self):
        """Test DAG layer forward pass with no_selector=True."""
        layer = DAGLayer(
            in_features=self.in_features,
            out_features=self.out_features,
            dag_depth=self.dag_depth,
            no_selector=True,
            _enable_taps=False,  # Disable taps to avoid wandb issues in tests
        )

        # Create test input
        x = torch.randn(self.batch_size, self.in_features)

        # Forward pass should work
        output = layer(x)

        # Output should have correct shape
        self.assertEqual(output.shape, (self.batch_size, self.out_features))

        # Should not contain NaN or inf
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_dag_layer_forward_without_no_selector(self):
        """Test DAG layer forward pass with no_selector=False."""
        layer = DAGLayer(
            in_features=self.in_features,
            out_features=self.out_features,
            dag_depth=self.dag_depth,
            no_selector=False,
            _enable_taps=False,  # Disable taps to avoid wandb issues in tests
        )

        # Create test input
        x = torch.randn(self.batch_size, self.in_features)

        # Forward pass should work
        output = layer(x)

        # Output should have correct shape
        self.assertEqual(output.shape, (self.batch_size, self.out_features))

        # Should not contain NaN or inf
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_dag_layer_different_outputs(self):
        """Test that no_selector=True and no_selector=False produce different outputs."""
        torch.manual_seed(42)  # For reproducibility

        # Create two identical layers except for no_selector
        layer_with_selector = DAGLayer(
            in_features=self.in_features,
            out_features=self.out_features,
            dag_depth=self.dag_depth,
            no_selector=False,
            _enable_taps=False,  # Disable taps to avoid wandb issues in tests
        )

        torch.manual_seed(42)  # Reset seed for identical initialization
        layer_no_selector = DAGLayer(
            in_features=self.in_features,
            out_features=self.out_features,
            dag_depth=self.dag_depth,
            no_selector=True,
            _enable_taps=False,  # Disable taps to avoid wandb issues in tests
        )

        # Create test input
        torch.manual_seed(123)
        x = torch.randn(self.batch_size, self.in_features)

        # Get outputs
        output_with_selector = layer_with_selector(x)
        output_no_selector = layer_no_selector(x)

        # Outputs should be different (in most cases)
        # We use allclose with a very small tolerance to handle numerical precision
        self.assertFalse(
            torch.allclose(output_with_selector, output_no_selector, atol=1e-6)
        )

    def test_single_layer_benchmark_no_selector_arg(self):
        """Test that single_layer_benchmark.py accepts --no-selector argument."""
        cmd = [
            sys.executable,
            "experiments/single_layer_benchmark.py",
            "--layer-type",
            "DAG",
            "--no-open-browser",
            "--operation",
            "add",
            "--seed",
            "42",
            "--max-iterations",
            "10",
            "--no-selector",
            "--help",  # Just check help to avoid running actual training
        ]

        try:
            # This should not raise an error if the argument is properly defined
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            # Help should contain the no-selector option
            self.assertIn("--no-selector", result.stdout)
        except subprocess.TimeoutExpired:
            # If help takes too long, that's also a failure
            self.fail("Command timed out - likely argument parsing issue")
        except FileNotFoundError:
            # Skip test if we can't find the script (e.g., in different test environment)
            self.skipTest("single_layer_benchmark.py not found")

    def test_dag_layer_training_vs_eval_modes(self):
        """Test that no_selector works in both training and eval modes."""
        layer = DAGLayer(
            in_features=self.in_features,
            out_features=self.out_features,
            dag_depth=self.dag_depth,
            no_selector=True,
            _enable_taps=False,  # Disable taps to avoid wandb issues in tests
        )

        # Create test input
        x = torch.randn(self.batch_size, self.in_features)

        # Test training mode
        layer.train()
        output_train = layer(x)
        self.assertEqual(output_train.shape, (self.batch_size, self.out_features))

        # Test eval mode
        layer.eval()
        with torch.no_grad():
            output_eval = layer(x)
        self.assertEqual(output_eval.shape, (self.batch_size, self.out_features))

    def test_dag_layer_reset_parameters_with_no_selector(self):
        """Test that reset_parameters works with no_selector=True."""
        layer = DAGLayer(
            in_features=self.in_features,
            out_features=self.out_features,
            dag_depth=self.dag_depth,
            no_selector=True,
        )

        # Get initial parameter value
        initial_param = layer.G_head.weight[0, 0].item()

        # Reset parameters
        layer.reset_parameters()

        # Parameter should have changed (with very high probability)
        new_param = layer.G_head.weight[0, 0].item()
        self.assertNotEqual(initial_param, new_param)

    def test_dag_layer_combined_with_freezing(self):
        """Test that no_selector works together with freeze_O_mul."""
        layer = DAGLayer(
            in_features=self.in_features,
            out_features=self.out_features,
            dag_depth=self.dag_depth,
            no_selector=True,
            freeze_O_mul=True,
            _enable_taps=False,  # Disable taps to avoid wandb issues in tests
        )

        # Should initialize successfully
        self.assertTrue(layer.no_selector)
        self.assertTrue(layer.freeze_O_mul)

        # Forward pass should work
        x = torch.randn(self.batch_size, self.in_features)
        output = layer(x)
        self.assertEqual(output.shape, (self.batch_size, self.out_features))


if __name__ == "__main__":
    unittest.main()
