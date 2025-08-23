#!/usr/bin/env python3
"""
Numerical stability tests for DAG layer to catch instabilities before training failures.
"""

import pytest
import torch
import torch.nn as nn

from stable_nalu.layer.dag import DAGLayer


class TestDAGNumericalStability:
    """Test suite for numerical stability of DAG layer operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.in_features = 4
        self.out_features = 1
        self.dag_depth = 3
        self.batch_size = 8

        # Create layer with deterministic initialization for testing
        self.layer = DAGLayer(
            self.in_features,
            self.out_features,
            self.dag_depth,
            _enable_taps=False,  # Disable debug output for clean testing
        )

        # Test inputs with various ranges to stress-test numerical behavior
        self.test_ranges = [
            torch.randn(self.batch_size, self.in_features) * 0.1,  # Small values
            torch.randn(self.batch_size, self.in_features) * 1.0,  # Normal values
            torch.randn(self.batch_size, self.in_features) * 10.0,  # Large values
            torch.tensor([[1e-6, 1e-6, 1e-6, 1e-6]] * self.batch_size),  # Tiny values
            torch.tensor(
                [[100.0, 100.0, 100.0, 100.0]] * self.batch_size
            ),  # Large values
        ]

    def test_forward_pass_finite_outputs(self):
        """Test that forward pass always produces finite outputs."""
        for test_input in self.test_ranges:
            output = self.layer(test_input)

            assert torch.isfinite(output).all(), (
                f"Non-finite output detected for input range {test_input.abs().max().item():.2e}. "
                f"Output: {output.flatten()[:5]}"
            )

    def test_gradient_computation_stability(self):
        """Test that gradients are finite and well-bounded."""
        for test_input in self.test_ranges:
            test_input.requires_grad_(True)

            # Forward pass
            output = self.layer(test_input)
            loss = output.sum()

            # Backward pass
            loss.backward()

            # Check input gradients
            assert torch.isfinite(
                test_input.grad
            ).all(), f"Non-finite input gradients for range {test_input.abs().max().item():.2e}"

            # Check parameter gradients
            for name, param in self.layer.named_parameters():
                if param.grad is not None:
                    assert torch.isfinite(
                        param.grad
                    ).all(), f"Non-finite gradient in parameter {name}"

                    # Check for reasonable gradient magnitudes (not too large)
                    grad_magnitude = param.grad.abs().max()
                    assert (
                        grad_magnitude < 1e6
                    ), f"Extremely large gradient in {name}: {grad_magnitude:.2e}"

    def test_sign_computation_scaling(self):
        """Test that sign computation produces reasonable values near ±1."""
        # Create a simple test case where we can predict the sign behavior
        test_input = torch.tensor([[2.0, 3.0, 1.0, 1.0]])

        # Set layer to evaluation mode for deterministic behavior
        self.layer.eval()

        with torch.no_grad():
            output = self.layer(test_input)

            # Access debug information if available
            if (
                hasattr(self.layer, "_debug_V_sign_new")
                and self.layer._debug_V_sign_new
            ):
                for i, sign_values in enumerate(self.layer._debug_V_sign_new):
                    sign_magnitude = sign_values.abs().max()

                    # Signs should be reasonably close to ±1, not tiny values
                    assert sign_magnitude > 0.1, (
                        f"Sign values too small at step {i}: max magnitude {sign_magnitude:.6f}. "
                        f"This suggests scaling issues in _compute_new_sign."
                    )

                    # Signs should not exceed 1 (due to tanh bounds)
                    assert (
                        sign_magnitude <= 1.0
                    ), f"Sign values exceed bounds at step {i}: max magnitude {sign_magnitude:.6f}"

    def test_magnitude_computation_bounds(self):
        """Test that magnitude computation respects bounds and doesn't explode."""
        for test_input in self.test_ranges:
            self.layer.eval()

            with torch.no_grad():
                output = self.layer(test_input)

                # Check that magnitudes stay within reasonable bounds
                if (
                    hasattr(self.layer, "_debug_V_mag_new")
                    and self.layer._debug_V_mag_new
                ):
                    for i, mag_values in enumerate(self.layer._debug_V_mag_new):
                        max_mag = mag_values.max()
                        min_mag = mag_values.min()

                        # Magnitudes should be positive
                        assert (
                            min_mag >= 0
                        ), f"Negative magnitude at step {i}: min {min_mag:.6f}"

                        # Magnitudes shouldn't explode beyond reasonable bounds
                        assert (
                            max_mag <= self.layer._mag_max
                        ), f"Magnitude exceeds _mag_max at step {i}: {max_mag:.2e} > {self.layer._mag_max:.2e}"

                        # Magnitudes shouldn't underflow below minimum
                        assert (
                            min_mag >= self.layer._mag_min
                        ), f"Magnitude below _mag_min at step {i}: {min_mag:.2e} < {self.layer._mag_min:.2e}"

    def test_gate_values_within_bounds(self):
        """Test that gate values stay within [0,1] bounds."""
        for test_input in self.test_ranges:
            # Test both training and eval modes
            for training_mode in [True, False]:
                self.layer.train(training_mode)

                with torch.no_grad():
                    output = self.layer(test_input)

                    # Check internal gate values if available
                    if hasattr(self.layer, "_last_G"):
                        G_values = self.layer._last_G

                        assert torch.all(G_values >= 0), (
                            f"Gate values below 0 in {'training' if training_mode else 'eval'} mode: "
                            f"min {G_values.min():.6f}"
                        )

                        assert torch.all(G_values <= 1), (
                            f"Gate values above 1 in {'training' if training_mode else 'eval'} mode: "
                            f"max {G_values.max():.6f}"
                        )

    def test_output_magnitude_reasonableness(self):
        """Test that outputs are in reasonable ranges given inputs."""
        # Test with known simple cases
        simple_inputs = [
            torch.tensor([[1.0, 1.0, 0.0, 0.0]]),  # Small positive
            torch.tensor([[-1.0, -1.0, 0.0, 0.0]]),  # Small negative
            torch.tensor([[10.0, 10.0, 0.0, 0.0]]),  # Larger positive
        ]

        self.layer.eval()

        for test_input in simple_inputs:
            with torch.no_grad():
                output = self.layer(test_input)

                input_magnitude = test_input.abs().max()
                output_magnitude = output.abs().max()

                # Output shouldn't be orders of magnitude different from input
                # (this is a heuristic - adjust based on expected behavior)
                ratio = output_magnitude / (input_magnitude + 1e-8)

                assert 1e-3 <= ratio <= 1e3, (
                    f"Output magnitude seems unreasonable. "
                    f"Input max: {input_magnitude:.3f}, Output max: {output_magnitude:.3f}, "
                    f"Ratio: {ratio:.2e}"
                )

    def test_no_nan_or_inf_intermediate_values(self):
        """Test that intermediate computations don't produce NaN or Inf."""
        # Patch the _is_nan method to be more strict for testing
        original_is_nan = self.layer._is_nan
        detected_issues = []

        def strict_is_nan(name, tensor):
            if not torch.isfinite(tensor).all():
                detected_issues.append(f"Non-finite values in {name}")
                return True
            return False

        self.layer._is_nan = strict_is_nan

        try:
            for test_input in self.test_ranges:
                detected_issues.clear()

                with torch.no_grad():
                    output = self.layer(test_input)

                assert (
                    len(detected_issues) == 0
                ), f"Detected numerical issues: {detected_issues}"
        finally:
            # Restore original method
            self.layer._is_nan = original_is_nan

    def test_consistent_eval_vs_training_mode(self):
        """Test that switching between training/eval modes doesn't cause instabilities."""
        test_input = torch.randn(self.batch_size, self.in_features)

        # Test mode transitions
        modes = [True, False, True, False]  # training, eval, training, eval

        outputs = []
        for mode in modes:
            self.layer.train(mode)

            with torch.no_grad():
                output = self.layer(test_input)
                outputs.append(output)

                # Each output should be finite
                assert torch.isfinite(
                    output
                ).all(), f"Non-finite output in {'training' if mode else 'eval'} mode"

        # Outputs should be reasonably stable (eval mode should be deterministic)
        eval_outputs = [outputs[1], outputs[3]]  # Both eval mode outputs
        output_diff = (eval_outputs[0] - eval_outputs[1]).abs().max()

        assert (
            output_diff < 1e-6
        ), f"Eval mode outputs not consistent: diff {output_diff:.2e}"


if __name__ == "__main__":
    pytest.main([__file__])
