#!/usr/bin/env python3
"""
Unit tests for _compute_aggregates method to verify core arithmetic aggregation logic.
"""

import torch

from stable_nalu.layer.dag import DAGLayer


class TestComputeAggregates:
    """Test suite for the _compute_aggregates method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.layer = DAGLayer(4, 1, 3, _enable_taps=False)
        self.layer.eval()

    def test_linear_domain_addition(self):
        """Test R_lin computation for addition: a + b."""
        # Setup: 2 + 3 = 5
        working_mag = torch.tensor([[2.0, 3.0, 0.0, 0.0]])  # magnitudes
        working_sign = torch.tensor([[1.0, 1.0, 0.0, 0.0]])  # signs
        O_step = torch.tensor([[1.0, 1.0, 0.0, 0.0]])  # addition selector: [+1, +1]

        R_lin, R_log = self.layer._compute_aggregates(working_mag, working_sign, O_step)

        # Expected: R_lin = 1*(1*2) + 1*(1*3) = 2 + 3 = 5
        expected_R_lin = 5.0
        assert (
            abs(R_lin.item() - expected_R_lin) < 1e-6
        ), f"R_lin={R_lin.item()}, expected={expected_R_lin}"

        # R_log should be log(2) + log(3) = log(6)
        expected_R_log = torch.log(torch.tensor(2.0)) + torch.log(torch.tensor(3.0))
        assert (
            abs(R_log.item() - expected_R_log.item()) < 1e-6
        ), f"R_log={R_log.item()}, expected={expected_R_log.item()}"

    def test_linear_domain_subtraction(self):
        """Test R_lin computation for subtraction: a - b."""
        # Setup: 5 - 2 = 3
        working_mag = torch.tensor([[5.0, 2.0, 0.0, 0.0]])
        working_sign = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        O_step = torch.tensor([[1.0, -1.0, 0.0, 0.0]])  # subtraction selector: [+1, -1]

        R_lin, R_log = self.layer._compute_aggregates(working_mag, working_sign, O_step)

        # Expected: R_lin = 1*(1*5) + (-1)*(1*2) = 5 - 2 = 3
        expected_R_lin = 3.0
        assert (
            abs(R_lin.item() - expected_R_lin) < 1e-6
        ), f"R_lin={R_lin.item()}, expected={expected_R_lin}"

        # R_log should be log(5) - log(2) = log(2.5)
        expected_R_log = torch.log(torch.tensor(5.0)) - torch.log(torch.tensor(2.0))
        assert (
            abs(R_log.item() - expected_R_log.item()) < 1e-6
        ), f"R_log={R_log.item()}, expected={expected_R_log.item()}"

    def test_linear_domain_negative_inputs(self):
        """Test R_lin with negative input values."""
        # Setup: (-2) + (-3) = -5
        working_mag = torch.tensor([[2.0, 3.0, 0.0, 0.0]])
        working_sign = torch.tensor([[-1.0, -1.0, 0.0, 0.0]])  # negative signs
        O_step = torch.tensor([[1.0, 1.0, 0.0, 0.0]])  # addition

        R_lin, R_log = self.layer._compute_aggregates(working_mag, working_sign, O_step)

        # Expected: R_lin = 1*(-1*2) + 1*(-1*3) = -2 + (-3) = -5
        expected_R_lin = -5.0
        assert (
            abs(R_lin.item() - expected_R_lin) < 1e-6
        ), f"R_lin={R_lin.item()}, expected={expected_R_lin}"

        # R_log should still be log(2) + log(3) = log(6) (signs lost)
        expected_R_log = torch.log(torch.tensor(2.0)) + torch.log(torch.tensor(3.0))
        assert (
            abs(R_log.item() - expected_R_log.item()) < 1e-6
        ), f"R_log={R_log.item()}, expected={expected_R_log.item()}"

    def test_linear_domain_mixed_signs(self):
        """Test R_lin with mixed positive/negative inputs."""
        # Setup: (-2) - (+3) = -5  (equivalent to (-2) + (-3))
        working_mag = torch.tensor([[2.0, 3.0, 0.0, 0.0]])
        working_sign = torch.tensor([[-1.0, 1.0, 0.0, 0.0]])  # mixed signs
        O_step = torch.tensor([[1.0, -1.0, 0.0, 0.0]])  # subtraction

        R_lin, R_log = self.layer._compute_aggregates(working_mag, working_sign, O_step)

        # Expected: R_lin = 1*(-1*2) + (-1)*(1*3) = -2 + (-3) = -5
        expected_R_lin = -5.0
        assert (
            abs(R_lin.item() - expected_R_lin) < 1e-6
        ), f"R_lin={R_lin.item()}, expected={expected_R_lin}"

    def test_log_domain_multiplication_setup(self):
        """Test R_log computation for multiplication setup: log(a) + log(b) = log(a*b)."""
        # Setup: 2 * 3 = 6 (in log space: log(2) + log(3) = log(6))
        working_mag = torch.tensor([[2.0, 3.0, 0.0, 0.0]])
        working_sign = torch.tensor([[1.0, 1.0, 0.0, 0.0]])  # signs will be lost
        O_step = torch.tensor([[1.0, 1.0, 0.0, 0.0]])  # multiplication selector

        R_lin, R_log = self.layer._compute_aggregates(working_mag, working_sign, O_step)

        # Expected: R_log = 1*log(2) + 1*log(3) = log(6)
        expected_R_log = torch.log(torch.tensor(6.0))
        assert (
            abs(R_log.item() - expected_R_log.item()) < 1e-6
        ), f"R_log={R_log.item()}, expected={expected_R_log.item()}"

    def test_log_domain_division_setup(self):
        """Test R_log computation for division setup: log(a) - log(b) = log(a/b)."""
        # Setup: 6 / 2 = 3 (in log space: log(6) - log(2) = log(3))
        working_mag = torch.tensor([[6.0, 2.0, 0.0, 0.0]])
        working_sign = torch.tensor([[1.0, 1.0, 0.0, 0.0]])  # signs will be lost
        O_step = torch.tensor([[1.0, -1.0, 0.0, 0.0]])  # division selector

        R_lin, R_log = self.layer._compute_aggregates(working_mag, working_sign, O_step)

        # Expected: R_log = 1*log(6) + (-1)*log(2) = log(6) - log(2) = log(3)
        expected_R_log = torch.log(torch.tensor(3.0))
        assert (
            abs(R_log.item() - expected_R_log.item()) < 1e-6
        ), f"R_log={R_log.item()}, expected={expected_R_log.item()}"

    def test_log_domain_loses_sign_information(self):
        """Test that R_log computation loses sign information completely."""
        # Setup: Same magnitudes, different signs
        working_mag = torch.tensor([[2.0, 3.0, 0.0, 0.0]])

        # Test 1: positive signs
        working_sign_pos = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        O_step = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        _, R_log_pos = self.layer._compute_aggregates(
            working_mag, working_sign_pos, O_step
        )

        # Test 2: negative signs
        working_sign_neg = torch.tensor([[-1.0, -1.0, 0.0, 0.0]])
        _, R_log_neg = self.layer._compute_aggregates(
            working_mag, working_sign_neg, O_step
        )

        # R_log should be identical regardless of signs (log domain loses sign info)
        assert (
            abs(R_log_pos.item() - R_log_neg.item()) < 1e-10
        ), f"R_log should be identical: pos={R_log_pos.item()}, neg={R_log_neg.item()}"

        # Both should equal log(2) + log(3) = log(6)
        expected_R_log = torch.log(torch.tensor(6.0))
        assert abs(R_log_pos.item() - expected_R_log.item()) < 1e-6

    def test_edge_case_zero_magnitudes(self):
        """Test behavior with zero magnitudes (should clamp to _mag_min)."""
        working_mag = torch.tensor([[0.0, 2.0, 0.0, 0.0]])  # First input is zero
        working_sign = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        O_step = torch.tensor([[1.0, 1.0, 0.0, 0.0]])

        R_lin, R_log = self.layer._compute_aggregates(working_mag, working_sign, O_step)

        # R_lin: 1*(1*0) + 1*(1*2) = 0 + 2 = 2
        expected_R_lin = 2.0
        assert abs(R_lin.item() - expected_R_lin) < 1e-6

        # R_log: log(clamp(0, min=_mag_min)) + log(2) = log(_mag_min) + log(2)
        expected_R_log = torch.log(torch.tensor(self.layer._mag_min)) + torch.log(
            torch.tensor(2.0)
        )
        assert abs(R_log.item() - expected_R_log.item()) < 1e-6

    def test_edge_case_very_small_magnitudes(self):
        """Test numerical stability with very small magnitudes."""
        small_val = 1e-10
        working_mag = torch.tensor([[small_val, 2.0, 0.0, 0.0]])
        working_sign = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        O_step = torch.tensor([[1.0, 1.0, 0.0, 0.0]])

        R_lin, R_log = self.layer._compute_aggregates(working_mag, working_sign, O_step)

        # Should not produce NaN or Inf
        assert torch.isfinite(R_lin).all(), f"R_lin not finite: {R_lin}"
        assert torch.isfinite(R_log).all(), f"R_log not finite: {R_log}"

        # R_log should clamp small value to _mag_min
        clamped_val = max(small_val, self.layer._mag_min)
        expected_R_log = torch.log(torch.tensor(clamped_val)) + torch.log(
            torch.tensor(2.0)
        )
        assert abs(R_log.item() - expected_R_log.item()) < 1e-6

    def test_edge_case_large_magnitudes(self):
        """Test behavior with large magnitudes."""
        large_val = 1e5
        working_mag = torch.tensor([[large_val, 2.0, 0.0, 0.0]])
        working_sign = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        O_step = torch.tensor([[1.0, 1.0, 0.0, 0.0]])

        R_lin, R_log = self.layer._compute_aggregates(working_mag, working_sign, O_step)

        # Should not produce NaN or Inf
        assert torch.isfinite(R_lin).all(), f"R_lin not finite: {R_lin}"
        assert torch.isfinite(R_log).all(), f"R_log not finite: {R_log}"

        # Values should be reasonable
        expected_R_lin = large_val + 2.0
        expected_R_log = torch.log(torch.tensor(large_val)) + torch.log(
            torch.tensor(2.0)
        )

        assert abs(R_lin.item() - expected_R_lin) < 1e-6
        assert abs(R_log.item() - expected_R_log.item()) < 1e-6

    def test_selector_weights_effect(self):
        """Test that O_step weights properly scale contributions."""
        working_mag = torch.tensor([[2.0, 3.0, 0.0, 0.0]])
        working_sign = torch.tensor([[1.0, 1.0, 0.0, 0.0]])

        # Test with partial weights
        O_step = torch.tensor([[0.5, 0.8, 0.0, 0.0]])  # Fractional selectors

        R_lin, R_log = self.layer._compute_aggregates(working_mag, working_sign, O_step)

        # Expected: R_lin = 0.5*(1*2) + 0.8*(1*3) = 1.0 + 2.4 = 3.4
        expected_R_lin = 0.5 * 2.0 + 0.8 * 3.0
        assert abs(R_lin.item() - expected_R_lin) < 1e-6

        # Expected: R_log = 0.5*log(2) + 0.8*log(3)
        expected_R_log = 0.5 * torch.log(torch.tensor(2.0)) + 0.8 * torch.log(
            torch.tensor(3.0)
        )
        assert abs(R_log.item() - expected_R_log.item()) < 1e-6

    def test_batch_processing(self):
        """Test that _compute_aggregates works correctly with batch dimensions."""
        batch_size = 3
        working_mag = torch.tensor(
            [
                [2.0, 3.0, 0.0, 0.0],  # Batch 0
                [1.0, 4.0, 0.0, 0.0],  # Batch 1
                [5.0, 2.0, 0.0, 0.0],  # Batch 2
            ]
        )
        working_sign = torch.tensor(
            [
                [1.0, 1.0, 0.0, 0.0],
                [1.0, -1.0, 0.0, 0.0],
                [-1.0, 1.0, 0.0, 0.0],
            ]
        )
        O_step = torch.tensor(
            [
                [1.0, 1.0, 0.0, 0.0],  # Addition
                [1.0, 1.0, 0.0, 0.0],  # Addition (but with negative sign)
                [1.0, -1.0, 0.0, 0.0],  # Subtraction
            ]
        )

        R_lin, R_log = self.layer._compute_aggregates(working_mag, working_sign, O_step)

        # Check shapes
        assert R_lin.shape == (batch_size, 1), f"R_lin shape: {R_lin.shape}"
        assert R_log.shape == (batch_size, 1), f"R_log shape: {R_log.shape}"

        # Check individual batch results
        # Batch 0: 1*(1*2) + 1*(1*3) = 5
        assert abs(R_lin[0].item() - 5.0) < 1e-6

        # Batch 1: 1*(1*1) + 1*(-1*4) = 1 - 4 = -3
        assert abs(R_lin[1].item() - (-3.0)) < 1e-6

        # Batch 2: 1*(-1*5) + (-1)*(1*2) = -5 - 2 = -7
        assert abs(R_lin[2].item() - (-7.0)) < 1e-6


def run_all_tests():
    """Run all tests and report results."""
    test_instance = TestComputeAggregates()
    test_instance.setup_method()

    tests = [
        test_instance.test_linear_domain_addition,
        test_instance.test_linear_domain_subtraction,
        test_instance.test_linear_domain_negative_inputs,
        test_instance.test_linear_domain_mixed_signs,
        test_instance.test_log_domain_multiplication_setup,
        test_instance.test_log_domain_division_setup,
        test_instance.test_log_domain_loses_sign_information,
        test_instance.test_edge_case_zero_magnitudes,
        test_instance.test_edge_case_very_small_magnitudes,
        test_instance.test_edge_case_large_magnitudes,
        test_instance.test_selector_weights_effect,
        test_instance.test_batch_processing,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            print(f"✅ {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__}: {e}")
            failed += 1

    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    if not success:
        exit(1)
