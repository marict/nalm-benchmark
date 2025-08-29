#!/usr/bin/env python3
"""
Comprehensive tests for sparsity error calculation in DAGLayer.
"""

import sys
import unittest
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

import numpy as np
import torch

from stable_nalu.layer.dag import DAGLayer


class TestSparsityError(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 4
        self.in_features = 2
        self.out_features = 1

    def create_dag_layer(self, dag_depth=1):
        """Helper to create DAG layer."""
        return DAGLayer(
            in_features=self.in_features,
            out_features=self.out_features,
            dag_depth=dag_depth,
            _enable_taps=False,  # Disable wandb logging in tests
        )

    def test_sparsity_error_dag_depth_constraint(self):
        """Test that sparsity error only works for dag_depth=1."""

        # Test dag_depth=1 (should work)
        layer1 = self.create_dag_layer(dag_depth=1)
        x = torch.randn(self.batch_size, self.in_features)
        _ = layer1(x)  # Initialize weights

        try:
            sparsity = layer1.calculate_sparsity_error("mul")
            self.assertIsInstance(sparsity, float)
            self.assertGreaterEqual(sparsity, 0.0)
            self.assertLessEqual(sparsity, 0.5)
        except ValueError:
            self.fail("dag_depth=1 should allow sparsity calculation")

        # Test dag_depth > 1 (should fail)
        for depth in [2, 3, 5]:
            layer = self.create_dag_layer(dag_depth=depth)
            x = torch.randn(self.batch_size, self.in_features)
            _ = layer(x)  # Initialize weights

            with self.assertRaises(ValueError) as cm:
                layer.calculate_sparsity_error("mul")

            self.assertIn(f"dag_depth={depth} > 1", str(cm.exception))
            self.assertIn("multiple valid solutions", str(cm.exception))

    def test_sparsity_error_operations(self):
        """Test that all operations work correctly."""

        layer = self.create_dag_layer(dag_depth=1)
        x = torch.randn(self.batch_size, self.in_features)
        _ = layer(x)  # Initialize weights

        operations = ["add", "sub", "mul", "div"]

        for op in operations:
            try:
                sparsity = layer.calculate_sparsity_error(op)
                self.assertIsInstance(sparsity, float)
                self.assertGreaterEqual(sparsity, 0.0)
                self.assertLessEqual(sparsity, 0.5)
            except Exception as e:
                self.fail(f"Operation '{op}' should work, but got: {e}")

        # Test invalid operation - should not raise error since operation is not used
        try:
            sparsity = layer.calculate_sparsity_error("invalid_op")
            self.assertIsInstance(sparsity, float)
            self.assertGreaterEqual(sparsity, 0.0)
            self.assertLessEqual(sparsity, 0.5)
        except Exception as e:
            self.fail(
                f"Invalid operation should not cause error since operation is not used: {e}"
            )

    def test_sparsity_error_mathematical_correctness(self):
        """Test the mathematical correctness of sparsity calculation."""

        layer = self.create_dag_layer(dag_depth=1)

        # Test Case 1: G = 0.0 (perfect log domain) → sparsity = 0.0
        with torch.no_grad():
            layer._last_train_raw_G = torch.zeros(self.batch_size, 1)
            layer._last_train_raw_G[0, 0] = 0.0  # Perfect log domain

        sparsity = layer.calculate_sparsity_error("mul")
        self.assertAlmostEqual(sparsity, 0.0, places=6)

        # Test Case 2: G = 1.0 (perfect linear domain) → sparsity = 0.0
        with torch.no_grad():
            layer._last_train_raw_G[0, 0] = 1.0  # Perfect linear domain

        sparsity = layer.calculate_sparsity_error("div")
        self.assertAlmostEqual(sparsity, 0.0, places=6)

        # Test Case 3: G = 0.5 (worst case, equally mixed) → sparsity = 0.5
        with torch.no_grad():
            layer._last_train_raw_G[0, 0] = 0.5  # Worst case mixing

        sparsity = layer.calculate_sparsity_error("add")
        self.assertAlmostEqual(sparsity, 0.5, places=6)

        # Test Case 4: G = 0.2 → sparsity = min(0.2, 0.8) = 0.2
        with torch.no_grad():
            layer._last_train_raw_G[0, 0] = 0.2  # Closer to log domain

        sparsity = layer.calculate_sparsity_error("sub")
        self.assertAlmostEqual(sparsity, 0.2, places=6)

        # Test Case 5: G = 0.9 → sparsity = min(0.9, 0.1) = 0.1
        with torch.no_grad():
            layer._last_train_raw_G[0, 0] = 0.9  # Closer to linear domain

        sparsity = layer.calculate_sparsity_error("mul")
        self.assertAlmostEqual(sparsity, 0.1, places=6)

    def test_sparsity_error_bounds(self):
        """Test that sparsity error is always in valid bounds."""

        layer = self.create_dag_layer(dag_depth=1)

        # Test with random G values
        np.random.seed(42)
        for _ in range(100):
            # Generate random G values in [-2, 2] range (will be clamped internally)
            g_val = np.random.uniform(-2, 2)

            with torch.no_grad():
                layer._last_train_raw_G = torch.zeros(self.batch_size, 1)
                layer._last_train_raw_G[0, 0] = float(g_val)

            sparsity = layer.calculate_sparsity_error("mul")

            # Sparsity should always be in [0, 0.5] range
            self.assertGreaterEqual(sparsity, 0.0, msg=f"G value: {g_val:.3f}")
            self.assertLessEqual(sparsity, 0.5, msg=f"G value: {g_val:.3f}")

    def test_sparsity_error_no_weights_available(self):
        """Test error handling when no weights are available."""

        layer = self.create_dag_layer(dag_depth=1)

        # Try to calculate sparsity without running forward pass
        with self.assertRaises(RuntimeError) as cm:
            layer.calculate_sparsity_error("mul")

        self.assertIn("Model hasn't been run yet", str(cm.exception))
        self.assertIn("Call forward() first", str(cm.exception))

    def test_sparsity_error_training_vs_eval_weights(self):
        """Test that sparsity uses the most recent weights (eval preferred)."""

        layer = self.create_dag_layer(dag_depth=1)
        x = torch.randn(self.batch_size, self.in_features)

        # Set up training weights
        layer.train()
        _ = layer(x)
        train_sparsity = layer.calculate_sparsity_error("mul")

        # Manually set different eval G weights
        with torch.no_grad():
            layer._last_eval_raw_G = torch.zeros(self.batch_size, 1)
            layer._last_eval_raw_G[0, 0] = 1.0  # Perfect linear domain

        # Should use eval G weights (perfect discrete → sparsity = 0.0)
        eval_sparsity = layer.calculate_sparsity_error("mul")
        self.assertAlmostEqual(eval_sparsity, 0.0, places=6)
        # Note: train_sparsity might also be 0.0, so we just verify eval works
        self.assertIsInstance(train_sparsity, float)

    def test_sparsity_error_with_more_than_two_inputs(self):
        """Test sparsity calculation with more input features."""

        # Create layer with 4 inputs
        layer = DAGLayer(in_features=4, out_features=1, dag_depth=1, _enable_taps=False)

        x = torch.randn(self.batch_size, 4)
        _ = layer(x)

        # G is always scalar for dag_depth=1, regardless of input count
        with torch.no_grad():
            layer._last_train_raw_G = torch.zeros(self.batch_size, 1)
            layer._last_train_raw_G[0, 0] = 0.3  # Some intermediate value

        sparsity = layer.calculate_sparsity_error("add")

        # Should be min(0.3, 0.7) = 0.3 regardless of input count
        expected_sparsity = min(0.3, 1.0 - 0.3)  # min(0.3, 0.7) = 0.3
        self.assertAlmostEqual(sparsity, expected_sparsity, places=6)


def run_integration_tests():
    """Run integration tests with actual training."""

    print("\n" + "=" * 60)
    print("SPARSITY ERROR INTEGRATION TESTS")
    print("=" * 60)

    print("\n1. Testing Sparsity During Actual Training...")

    # Create a simple training setup
    layer = DAGLayer(
        in_features=2,
        out_features=1,
        dag_depth=1,
        _enable_taps=False,
        freeze_O_mul=True,  # Use frozen weights for predictable behavior
    )

    # Create simple multiplication training data
    x_train = torch.tensor([[1.0, 2.0], [2.0, 3.0], [0.5, 4.0]])
    y_train = torch.tensor([[2.0], [6.0], [2.0]])  # x1 * x2

    optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)

    print("   Training for 5 steps and tracking sparsity...")
    sparsity_values = []

    for step in range(5):
        optimizer.zero_grad()

        output = layer(x_train)
        loss = torch.nn.MSELoss()(output, y_train)
        loss.backward()
        optimizer.step()

        try:
            sparsity = layer.calculate_sparsity_error("mul")
            sparsity_values.append(sparsity)
            print(f"     Step {step}: Loss={loss.item():.4f}, Sparsity={sparsity:.6f}")
        except Exception as e:
            print(f"     Step {step}: Sparsity calculation failed: {e}")

    if sparsity_values:
        print(f"   ✅ Sparsity tracking successful")
        print(
            f"   ✅ All values in valid range [0, 0.5]: {all(0 <= s <= 0.5 for s in sparsity_values)}"
        )

        # Check if sparsity changes (not hardcoded)
        sparsity_range = max(sparsity_values) - min(sparsity_values)
        if sparsity_range > 1e-6:
            print(
                f"   ✅ Sparsity values change during training (range: {sparsity_range:.6f})"
            )
        else:
            print(
                f"   ⚠️  Sparsity values don't change much (range: {sparsity_range:.6f})"
            )
    else:
        print(f"   ❌ No sparsity values calculated")

    print("\n2. Testing Sparsity with Different Operations...")

    operations = ["add", "sub", "mul", "div"]
    for op in operations:
        try:
            sparsity = layer.calculate_sparsity_error(op)
            print(f"   {op.upper()}: {sparsity:.6f} ✅")
        except Exception as e:
            print(f"   {op.upper()}: ERROR - {e} ❌")

    print("\n3. Testing Edge Cases...")

    # Test with different dag_depth values
    for depth in [1, 2, 3]:
        try:
            test_layer = DAGLayer(
                in_features=2, out_features=1, dag_depth=depth, _enable_taps=False
            )
            x = torch.randn(2, 2)
            _ = test_layer(x)

            sparsity = test_layer.calculate_sparsity_error("mul")
            if depth == 1:
                print(f"   dag_depth={depth}: {sparsity:.6f} ✅")
            else:
                print(
                    f"   dag_depth={depth}: Should have failed but got {sparsity:.6f} ❌"
                )
        except ValueError as e:
            if depth > 1:
                print(f"   dag_depth={depth}: Correctly rejected ✅")
            else:
                print(f"   dag_depth={depth}: Unexpected error: {e} ❌")
        except Exception as e:
            print(f"   dag_depth={depth}: Unexpected error: {e} ❌")

    print(f"\n{'='*60}")
    print("INTEGRATION TESTS COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Run unit tests
    print("Running Sparsity Error Unit Tests...")
    unittest.main(argv=[""], exit=False, verbosity=2)

    # Run integration tests
    run_integration_tests()
