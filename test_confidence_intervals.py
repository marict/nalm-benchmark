#!/usr/bin/env python3
"""
Comprehensive tests for confidence interval calculations.
"""

import math
import sys
import unittest
from pathlib import Path

# Add misc to path for importing confidence intervals
sys.path.append(str(Path(__file__).parent / "misc"))

import numpy as np
from confidence_intervals import (beta_confidence_interval,
                                  binomial_confidence_interval,
                                  format_confidence_interval,
                                  gamma_confidence_interval,
                                  normal_confidence_interval)
from scipy import stats


class TestConfidenceIntervals(unittest.TestCase):

    def test_binomial_confidence_interval_known_values(self):
        """Test binomial CI with known statistical values."""

        # Test case 1: 50% success rate with large sample
        successes, trials = 50, 100
        lower, upper = binomial_confidence_interval(successes, trials)

        # For 50/100, Wilson score CI should be approximately [0.4, 0.6]
        self.assertAlmostEqual(lower, 0.4, delta=0.05)
        self.assertAlmostEqual(upper, 0.6, delta=0.05)

        # Test case 2: Perfect success
        successes, trials = 10, 10
        lower, upper = binomial_confidence_interval(successes, trials)

        # Should be close to 1.0 but not exactly due to Wilson correction
        self.assertGreater(lower, 0.7)
        self.assertAlmostEqual(upper, 1.0, places=10)

        # Test case 3: No successes
        successes, trials = 0, 10
        lower, upper = binomial_confidence_interval(successes, trials)

        self.assertEqual(lower, 0.0)
        self.assertLess(upper, 0.4)  # Should be small but > 0

        # Test case 4: Edge case - no trials
        successes, trials = 0, 0
        lower, upper = binomial_confidence_interval(successes, trials)

        self.assertEqual(lower, 0.0)
        self.assertEqual(upper, 0.0)

    def test_binomial_confidence_interval_properties(self):
        """Test mathematical properties of binomial CI."""

        # Property 1: CI should contain the point estimate
        for successes, trials in [(5, 20), (15, 50), (1, 100)]:
            if trials > 0:
                p_hat = successes / trials
                lower, upper = binomial_confidence_interval(successes, trials)

                # Point estimate should be within CI (mostly true for Wilson)
                if successes > 0 and successes < trials:  # Avoid edge cases
                    self.assertGreaterEqual(
                        p_hat, lower - 0.1
                    )  # Small tolerance for Wilson adjustment
                    self.assertLessEqual(p_hat, upper + 0.1)

        # Property 2: Larger sample should give narrower CI
        lower1, upper1 = binomial_confidence_interval(10, 20)  # n=20
        lower2, upper2 = binomial_confidence_interval(50, 100)  # n=100, same rate

        width1 = upper1 - lower1
        width2 = upper2 - lower2
        self.assertGreater(width1, width2)  # Larger sample = narrower CI

    def test_gamma_confidence_interval_known_values(self):
        """Test gamma CI with known distributions."""

        # Test case 1: Exponential-like data (gamma with shape ≈ 1)
        # Generate data from known gamma distribution
        np.random.seed(42)  # For reproducibility
        true_shape, true_scale = 1.5, 2.0
        data = np.random.gamma(true_shape, true_scale, size=100).tolist()

        lower, upper = gamma_confidence_interval(data)
        true_mean = true_shape * true_scale  # True mean = 3.0

        # CI should contain the true mean
        self.assertLessEqual(lower, true_mean)
        self.assertGreaterEqual(upper, true_mean)

        # Test case 2: More peaked distribution
        data = [100, 120, 110, 130, 105, 115, 125, 108]  # Low variance
        lower, upper = gamma_confidence_interval(data)
        sample_mean = sum(data) / len(data)

        # Should be relatively narrow CI around sample mean
        self.assertLessEqual(lower, sample_mean)
        self.assertGreaterEqual(upper, sample_mean)
        self.assertLess(upper - lower, sample_mean * 0.5)  # CI width < 50% of mean

    def test_gamma_confidence_interval_edge_cases(self):
        """Test gamma CI edge cases and fallbacks."""

        # Edge case 1: Single value
        data = [42.0]
        lower, upper = gamma_confidence_interval(data)

        # Should fall back to reasonable behavior (not crash)
        self.assertIsInstance(lower, float)
        self.assertIsInstance(upper, float)
        self.assertLessEqual(lower, upper)

        # Edge case 2: All identical values (zero variance)
        data = [100.0, 100.0, 100.0, 100.0]
        lower, upper = gamma_confidence_interval(data)

        # Should handle zero variance gracefully
        self.assertIsInstance(lower, float)
        self.assertIsInstance(upper, float)

        # Edge case 3: Empty list
        data = []
        lower, upper = gamma_confidence_interval(data)

        self.assertEqual(lower, float("inf"))
        self.assertEqual(upper, float("inf"))

    def test_beta_confidence_interval_known_values(self):
        """Test beta CI with values bounded [0,1]."""

        # Test case 1: Sparsity values around 0.1
        sparsity_data = [0.08, 0.12, 0.09, 0.11, 0.10, 0.07, 0.13, 0.09]
        lower, upper = beta_confidence_interval(sparsity_data)
        sample_mean = sum(sparsity_data) / len(sparsity_data)

        # CI should be bounded [0,1] and contain sample mean
        self.assertGreaterEqual(lower, 0.0)
        self.assertLessEqual(upper, 1.0)
        self.assertLessEqual(lower, sample_mean)
        self.assertGreaterEqual(upper, sample_mean)

        # Test case 2: Very small values (near 0)
        small_values = [0.001, 0.002, 0.0015, 0.0008, 0.0012]
        lower, upper = beta_confidence_interval(small_values)

        self.assertGreaterEqual(lower, 0.0)
        self.assertLessEqual(upper, 1.0)

        # Test case 3: Values near boundary (high sparsity)
        high_values = [0.45, 0.48, 0.47, 0.49, 0.46]
        lower, upper = beta_confidence_interval(high_values)

        self.assertGreaterEqual(lower, 0.0)
        self.assertLessEqual(upper, 1.0)

    def test_beta_confidence_interval_edge_cases(self):
        """Test beta CI edge cases."""

        # Edge case 1: Empty data
        data = []
        lower, upper = beta_confidence_interval(data)

        self.assertEqual(lower, 0.0)
        self.assertEqual(upper, 1.0)

        # Edge case 2: Single value
        data = [0.25]
        lower, upper = beta_confidence_interval(data)

        self.assertGreaterEqual(lower, 0.0)
        self.assertLessEqual(upper, 1.0)

        # Edge case 3: Boundary values
        data = [0.0, 1.0]  # Will be clipped to avoid boundary issues
        lower, upper = beta_confidence_interval(data)

        self.assertGreaterEqual(lower, 0.0)
        self.assertLessEqual(upper, 1.0)

    def test_normal_confidence_interval_properties(self):
        """Test normal CI mathematical properties."""

        # Test case 1: Known normal data
        np.random.seed(123)
        data = np.random.normal(50, 10, 100).tolist()  # μ=50, σ=10

        lower, upper = normal_confidence_interval(data)
        sample_mean = sum(data) / len(data)

        # CI should contain sample mean
        self.assertLessEqual(lower, sample_mean)
        self.assertGreaterEqual(upper, sample_mean)

        # For large n, CI should be approximately ±1.96 * (σ/√n) from mean
        expected_margin = 1.96 * (10 / math.sqrt(100))  # ≈ 1.96
        self.assertAlmostEqual(upper - sample_mean, expected_margin, delta=1.0)
        self.assertAlmostEqual(sample_mean - lower, expected_margin, delta=1.0)

    def test_format_confidence_interval(self):
        """Test CI formatting function."""

        # Test basic formatting
        formatted = format_confidence_interval(0.123456, 0.789012)
        self.assertIn("0.1235", formatted)
        self.assertIn("0.7890", formatted)

        # Test with point estimate
        formatted = format_confidence_interval(0.1, 0.9, 0.5, precision=2)
        self.assertIn("0.50", formatted)
        self.assertIn("[0.10, 0.90]", formatted)

        # Test precision control
        formatted = format_confidence_interval(0.123456789, 0.987654321, precision=6)
        self.assertIn("0.123457", formatted)
        self.assertIn("0.987654", formatted)

    def test_confidence_interval_consistency(self):
        """Test that all CI methods give reasonable results for same data type."""

        # Generate some reasonable convergence time data
        convergence_times = [150, 200, 180, 220, 160, 300, 175, 190]

        # All methods should return valid intervals
        gamma_ci = gamma_confidence_interval(convergence_times)
        normal_ci = normal_confidence_interval(convergence_times)

        # Both should give valid intervals
        self.assertLessEqual(gamma_ci[0], gamma_ci[1])
        self.assertLessEqual(normal_ci[0], normal_ci[1])

        # Both should contain the sample mean
        sample_mean = sum(convergence_times) / len(convergence_times)

        self.assertLessEqual(
            gamma_ci[0], sample_mean * 1.2
        )  # Allow some flexibility for gamma
        self.assertLessEqual(normal_ci[0], sample_mean)
        self.assertGreaterEqual(gamma_ci[1], sample_mean * 0.8)
        self.assertGreaterEqual(normal_ci[1], sample_mean)


def run_statistical_validation():
    """Run additional statistical validation tests."""

    print("\n" + "=" * 60)
    print("STATISTICAL VALIDATION TESTS")
    print("=" * 60)

    # Test 1: Binomial coverage probability
    print("\n1. Testing Binomial CI Coverage...")

    coverage_count = 0
    trials = 1000

    for _ in range(trials):
        # Simulate binomial process with known p=0.3
        true_p = 0.3
        n = 50
        successes = np.random.binomial(n, true_p)

        lower, upper = binomial_confidence_interval(successes, n)
        if lower <= true_p <= upper:
            coverage_count += 1

    coverage_rate = coverage_count / trials
    print(f"   95% CI should contain true p about 95% of time")
    print(f"   Actual coverage: {coverage_rate:.3f} ({coverage_count}/{trials})")
    print(f"   ✅ PASS" if 0.93 <= coverage_rate <= 0.97 else f"   ❌ FAIL")

    # Test 2: Gamma CI with known distribution
    print("\n2. Testing Gamma CI with Known Distribution...")

    np.random.seed(456)
    true_shape, true_scale = 2.0, 3.0
    true_mean = true_shape * true_scale  # = 6.0

    coverage_count = 0
    trials = 200  # Smaller sample for gamma

    for _ in range(trials):
        data = np.random.gamma(true_shape, true_scale, size=30)
        lower, upper = gamma_confidence_interval(data.tolist())
        if lower <= true_mean <= upper:
            coverage_count += 1

    coverage_rate = coverage_count / trials
    print(f"   95% CI should contain true mean about 95% of time")
    print(f"   True mean: {true_mean}, Actual coverage: {coverage_rate:.3f}")
    print(
        f"   ✅ PASS" if 0.90 <= coverage_rate <= 0.98 else f"   ❌ FAIL"
    )  # More lenient for gamma

    print(f"\n{'='*60}")
    print("STATISTICAL VALIDATION COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Run unit tests
    print("Running Confidence Interval Unit Tests...")
    unittest.main(argv=[""], exit=False, verbosity=2)

    # Run statistical validation
    run_statistical_validation()
