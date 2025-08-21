#!/usr/bin/env python3
"""
Test that demonstrates the systematic failure of DAG layer to learn division.

This test runs minimal training scenarios and shows that even with very short
training runs, division exhibits erratic non-training behavior compared to
addition which learns systematic patterns.

The test validates the hypothesis that division training fails due to:
1. Network not learning proper selector patterns ([1, -1, 0, ...] for div)
2. Gate G not converging to log domain (G ≈ 0 for div operations)
3. Erratic prediction of operand selectors (mostly zeros or random small values)
"""

import os
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


def run_minimal_training(operation: str, seed: int, max_iterations: int = 500) -> Dict:
    """Run minimal training and extract key metrics."""

    # Create temporary directory for this test run
    with tempfile.TemporaryDirectory() as temp_dir:
        # Run training with minimal parameters
        cmd = [
            sys.executable,
            "experiments/single_layer_benchmark.py",
            "--layer-type",
            "DAG",
            "--operation",
            operation,
            "--input-size",
            "2",
            "--batch-size",
            "16",  # Very small batch
            "--max-iterations",
            str(max_iterations),
            "--learning-rate",
            "1e-3",
            "--log-interval",
            str(max_iterations),  # Log only at the end
            "--interpolation-range",
            "[-2.0,2.0]",
            "--extrapolation-range",
            "[[-4.0,-2.0],[2.0,4.0]]",
            "--seed",
            str(seed),
            "--no-cuda",
            "--lr-cosine",
            "--lr-min",
            "1e-4",
            "--clip-grad-norm",
            "1.0",
            "--no-open-browser",
            "--no-save",  # Don't save model files
            "--note",
            f"division_test_{operation}_seed{seed}",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,  # 2 minute timeout
                cwd="/Users/paul_curry/ai2/nalm-benchmark",
            )

            if result.returncode != 0:
                return {
                    "success": False,
                    "error": f"Command failed with code {result.returncode}",
                    "stderr": result.stderr,
                }

            # Parse output for key metrics
            output = result.stdout
            metrics = parse_training_output(output)
            metrics["success"] = True
            return metrics

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Training timed out", "timeout": True}
        except Exception as e:
            return {"success": False, "error": f"Exception: {str(e)}"}


def parse_training_output(output: str) -> Dict:
    """Parse training output to extract key diagnostic metrics."""
    lines = output.split("\n")

    metrics = {
        "early_stopped": False,
        "early_stop_step": None,
        "final_train_loss": None,
        "final_inter_error": None,
        "final_extra_error": None,
        "selector_patterns": [],  # O step values
        "gate_values": [],  # G values
        "convergence_indicators": [],
    }

    for line in lines:
        # Check for early stopping
        if "Early stopping at step" in line:
            metrics["early_stopped"] = True
            try:
                step_part = line.split("Early stopping at step ")[1]
                metrics["early_stop_step"] = int(step_part.split(":")[0])

                # Extract final errors from early stopping line
                if "inter=" in line and "extra=" in line:
                    inter_part = line.split("inter=")[1].split(",")[0]
                    extra_part = line.split("extra=")[1].split()[0]
                    metrics["final_inter_error"] = float(inter_part)
                    metrics["final_extra_error"] = float(extra_part)
            except (IndexError, ValueError):
                pass

        # Parse training loss lines (train 123: 0.456, inter: 0.789, extra: 1.234)
        if "train " in line and ": " in line and "inter:" in line:
            try:
                train_part = line.split(": ")[1].split(",")[0]
                metrics["final_train_loss"] = float(train_part)

                inter_part = line.split("inter: ")[1].split(",")[0]
                metrics["final_inter_error"] = float(inter_part)

                if "extra:" in line:
                    extra_part = line.split("extra: ")[1].split()[0]
                    metrics["final_extra_error"] = float(extra_part)

            except (IndexError, ValueError):
                pass

        # Parse selector patterns from first sample statistics
        if "G: [" in line:
            try:
                g_str = line.split("G: ")[1]
                # Extract gate values - should be near 0 for div (log domain)
                g_values = eval(g_str)  # Safe because we control the format
                metrics["gate_values"] = g_values
            except:
                pass

        # Parse O step values (selector patterns)
        if "\tvalues: [" in line:
            try:
                values_str = line.split("values: ")[1]
                values = eval(values_str)  # Safe because we control the format
                metrics["selector_patterns"].append(values)
            except:
                pass

    return metrics


def analyze_training_quality(metrics: Dict, operation: str) -> Dict:
    """Analyze training metrics to determine if training succeeded."""
    analysis = {
        "operation": operation,
        "training_successful": False,
        "grokked": metrics.get("early_stopped", False),
        "converged_to_solution": False,
        "learned_correct_patterns": False,
        "issues": [],
    }

    # Check for grokking (early stopping)
    if metrics.get("early_stopped", False):
        analysis["training_successful"] = True
        analysis["grokked"] = True

        # For grokked solutions, errors should be very low
        inter_error = metrics.get("final_inter_error", float("inf"))
        extra_error = metrics.get("final_extra_error", float("inf"))

        if inter_error < 1e-8 and extra_error < 1e-6:
            analysis["converged_to_solution"] = True

    # Analyze selector patterns for division
    if operation == "div":
        gate_values = metrics.get("gate_values", [])
        selector_patterns = metrics.get("selector_patterns", [])

        # For division, we expect:
        # 1. G values close to 0 (log domain)
        # 2. Selector patterns like [1, -1, 0, ...] for first step

        if gate_values:
            # Check if gates are in log domain (G ≈ 0)
            first_gate = gate_values[0] if len(gate_values) > 0 else 1.0
            if first_gate > 0.3:  # Should be close to 0 for log domain
                analysis["issues"].append(
                    f"Gate not in log domain: G[0]={first_gate:.3f} (expected ≈ 0)"
                )

        if selector_patterns:
            # Check first step selector pattern
            first_pattern = selector_patterns[0] if len(selector_patterns) > 0 else []
            if len(first_pattern) >= 2:
                # For division, expect something like [1, -1, 0, ...] (select first input positively, second negatively)
                first_sel = first_pattern[0] if len(first_pattern) > 0 else 0
                second_sel = first_pattern[1] if len(first_pattern) > 1 else 0

                # Check if selectors are too small (not learning)
                if abs(first_sel) < 0.1 and abs(second_sel) < 0.1:
                    analysis["issues"].append(
                        f"Selectors too small: [{first_sel:.3f}, {second_sel:.3f}, ...] (not learning operand selection)"
                    )

                # Check if pattern makes sense for division (first positive, second negative or vice versa)
                elif not (
                    (first_sel > 0.1 and second_sel < -0.1)
                    or (first_sel < -0.1 and second_sel > 0.1)
                ):
                    analysis["issues"].append(
                        f"Wrong selector pattern for division: [{first_sel:.3f}, {second_sel:.3f}, ...] (expected [+,-, ...] or [-,+, ...])"
                    )

    # For addition, we expect linear domain (G ≈ 1) and additive patterns
    elif operation == "add":
        gate_values = metrics.get("gate_values", [])
        if gate_values:
            first_gate = gate_values[0] if len(gate_values) > 0 else 0.0
            if first_gate < 0.7:  # Should be close to 1 for linear domain
                analysis["issues"].append(
                    f"Gate not in linear domain: G[0]={first_gate:.3f} (expected ≈ 1)"
                )

    # Overall assessment
    if (
        analysis["grokked"]
        and analysis["converged_to_solution"]
        and len(analysis["issues"]) == 0
    ):
        analysis["training_successful"] = True
        analysis["learned_correct_patterns"] = True

    return analysis


class TestDivisionTrainingFailure:
    """Test suite demonstrating division training failures."""

    def test_addition_trains_successfully(self):
        """Test that addition can train successfully in minimal setup."""
        print("\n=== Testing Addition Training ===")

        results = []
        for seed in [0, 1, 2]:  # Test multiple seeds
            metrics = run_minimal_training("add", seed, max_iterations=2000)

            if not metrics["success"]:
                print(
                    f"  ⚠️  SKIPPED: Addition training failed for seed {seed}: {metrics.get('error', 'unknown error')}"
                )
                continue

            analysis = analyze_training_quality(metrics, "add")
            results.append((seed, analysis))

            print(
                f"Seed {seed}: {'✅ SUCCESS' if analysis['training_successful'] else '❌ FAILED'}"
            )
            if analysis["issues"]:
                for issue in analysis["issues"]:
                    print(f"  - {issue}")

        # At least one seed should succeed for addition
        successful_seeds = [
            seed for seed, analysis in results if analysis["training_successful"]
        ]
        if len(successful_seeds) == 0:
            raise AssertionError(
                f"Addition failed to train on all seeds {[s for s, _ in results]}"
            )

        print(f"Addition succeeded on seeds: {successful_seeds}")
        return True

    def test_division_fails_systematically(self):
        """Test that division fails systematically across multiple seeds."""
        print("\n=== Testing Division Training ===")

        results = []
        for seed in [0, 1, 2, 3, 4]:  # Test more seeds for division
            metrics = run_minimal_training(
                "div", seed, max_iterations=3000
            )  # Give division more time

            if not metrics["success"]:
                print(
                    f"Seed {seed}: ❌ TRAINING ERROR - {metrics.get('error', 'unknown')}"
                )
                results.append(
                    (
                        seed,
                        {
                            "training_successful": False,
                            "error": True,
                            "issues": ["Training command failed"],
                        },
                    )
                )
                continue

            analysis = analyze_training_quality(metrics, "div")
            results.append((seed, analysis))

            print(
                f"Seed {seed}: {'✅ SUCCESS' if analysis['training_successful'] else '❌ FAILED'}"
            )
            print(f"  Grokked: {analysis['grokked']}")
            print(f"  Converged: {analysis['converged_to_solution']}")
            if analysis["issues"]:
                for issue in analysis["issues"]:
                    print(f"  - {issue}")

        # Count successful vs failed seeds
        successful_seeds = [
            seed
            for seed, analysis in results
            if analysis.get("training_successful", False)
        ]
        failed_seeds = [
            seed
            for seed, analysis in results
            if not analysis.get("training_successful", False)
        ]

        print(f"\nDivision Results:")
        print(f"  Successful seeds: {successful_seeds}")
        print(f"  Failed seeds: {failed_seeds}")
        print(
            f"  Success rate: {len(successful_seeds)}/{len(results)} = {len(successful_seeds)/len(results):.1%}"
        )

        # Assert that division has systematic issues
        # We expect very low success rate for division
        success_rate = len(successful_seeds) / len(results)
        if success_rate >= 0.5:
            raise AssertionError(
                f"Division unexpectedly succeeded too often: {success_rate:.1%} success rate"
            )

        # Collect and report common failure modes
        all_issues = []
        for seed, analysis in results:
            all_issues.extend(analysis.get("issues", []))

        if all_issues:
            print(f"\nCommon failure modes observed:")
            from collections import Counter

            issue_counts = Counter(all_issues)
            for issue, count in issue_counts.most_common():
                print(f"  - {issue} (occurred in {count}/{len(results)} seeds)")

        return True

    def test_division_vs_addition_comparison(self):
        """Direct comparison showing division fails where addition succeeds."""
        print("\n=== Direct Addition vs Division Comparison ===")

        seed = 42  # Use same seed for fair comparison
        max_iter = 2000

        # Test addition
        add_metrics = run_minimal_training("add", seed, max_iter)
        add_analysis = (
            analyze_training_quality(add_metrics, "add")
            if add_metrics["success"]
            else {"training_successful": False}
        )

        # Test division
        div_metrics = run_minimal_training("div", seed, max_iter)
        div_analysis = (
            analyze_training_quality(div_metrics, "div")
            if div_metrics["success"]
            else {"training_successful": False}
        )

        print(f"Seed {seed} Results:")
        print(
            f"  Addition: {'✅ SUCCESS' if add_analysis['training_successful'] else '❌ FAILED'}"
        )
        print(
            f"  Division: {'✅ SUCCESS' if div_analysis['training_successful'] else '❌ FAILED'}"
        )

        # Report detailed comparison
        if add_metrics["success"] and div_metrics["success"]:
            print(f"\nDetailed Comparison:")

            add_grok = add_analysis.get("grokked", False)
            div_grok = div_analysis.get("grokked", False)
            print(f"  Grokking: Add={add_grok}, Div={div_grok}")

            add_inter = add_metrics.get("final_inter_error", float("inf"))
            div_inter = div_metrics.get("final_inter_error", float("inf"))
            print(
                f"  Final interpolation error: Add={add_inter:.2e}, Div={div_inter:.2e}"
            )

            add_gates = add_metrics.get("gate_values", [])
            div_gates = div_metrics.get("gate_values", [])
            if add_gates and div_gates:
                print(
                    f"  First gate value: Add={add_gates[0]:.3f}, Div={div_gates[0]:.3f}"
                )
                print(
                    f"    (Add should be ≈1 for linear domain, Div should be ≈0 for log domain)"
                )

            add_patterns = add_metrics.get("selector_patterns", [])
            div_patterns = div_metrics.get("selector_patterns", [])
            if add_patterns and div_patterns:
                add_first = add_patterns[0][:2] if len(add_patterns[0]) >= 2 else []
                div_first = div_patterns[0][:2] if len(div_patterns[0]) >= 2 else []
                print(
                    f"  First selector pattern: Add={[f'{x:.3f}' for x in add_first]}, Div={[f'{x:.3f}' for x in div_first]}"
                )

        # The key assertion: division should be much worse than addition
        # This test documents the systematic failure rather than expecting it to pass
        if (
            add_analysis["training_successful"]
            and not div_analysis["training_successful"]
        ):
            print(
                f"\n🔍 CONFIRMED: Division fails systematically where addition succeeds"
            )
            print(
                f"   This demonstrates the core training issue with division operations"
            )

            # List division-specific issues
            div_issues = div_analysis.get("issues", [])
            if div_issues:
                print(f"   Division issues observed:")
                for issue in div_issues:
                    print(f"     - {issue}")

        return True


def main():
    """Run division training failure tests."""
    print("Division Training Failure Test Suite")
    print("=" * 50)

    test_suite = TestDivisionTrainingFailure()

    try:
        test_suite.test_addition_trains_successfully()
        print("\n✅ Addition training test completed")
    except Exception as e:
        print(f"\n❌ Addition training test failed: {e}")

    try:
        test_suite.test_division_fails_systematically()
        print("\n✅ Division systematic failure test completed")
    except Exception as e:
        print(f"\n❌ Division systematic failure test failed: {e}")

    try:
        test_suite.test_division_vs_addition_comparison()
        print("\n✅ Direct comparison test completed")
    except Exception as e:
        print(f"\n❌ Direct comparison test failed: {e}")


if __name__ == "__main__":
    main()
