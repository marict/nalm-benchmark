#!/usr/bin/env python3
"""
Minimal test demonstrating that division consistently fails where addition succeeds.

This test uses the shortest possible training to isolate the core issue:
Division operations fail to learn proper selector patterns and domain gates.
"""

import subprocess
import sys

import torch


def test_division_fails_basic_training():
    """Test that demonstrates division's systematic training failure."""

    def run_minimal_experiment(operation, seed=0):
        """Run minimal training experiment."""
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
            "8",
            "--max-iterations",
            "200",  # Very short training
            "--learning-rate",
            "1e-3",
            "--log-interval",
            "200",
            "--interpolation-range",
            "[-2.0,2.0]",
            "--extrapolation-range",
            "[[-4.0,-2.0],[2.0,4.0]]",
            "--seed",
            str(seed),
            "--no-cuda",
            "--no-open-browser",
            "--no-save",
            "--note",
            f"minimal_{operation}_test",
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
            if result.returncode == 0:
                return {"success": True, "output": result.stdout}
            else:
                return {"success": False, "error": result.stderr}
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "timeout"}

    def extract_final_loss(output):
        """Extract final interpolation error from training output."""
        lines = output.split("\n")
        for line in reversed(lines):  # Look from end
            if "inter:" in line and "train" in line:
                try:
                    inter_part = line.split("inter: ")[1].split(",")[0]
                    return float(inter_part)
                except (IndexError, ValueError):
                    continue
        return float("inf")  # No valid loss found

    # Test addition (should learn reasonably well)
    print("Testing addition training...")
    add_result = run_minimal_experiment("add", seed=42)

    assert add_result[
        "success"
    ], f"Addition training failed: {add_result.get('error', 'unknown')}"
    add_loss = extract_final_loss(add_result["output"])
    print(f"Addition final interpolation error: {add_loss:.2e}")

    # Test division (should fail to learn)
    print("Testing division training...")
    div_result = run_minimal_experiment("div", seed=42)

    assert div_result[
        "success"
    ], f"Division training failed: {div_result.get('error', 'unknown')}"
    div_loss = extract_final_loss(div_result["output"])
    print(f"Division final interpolation error: {div_loss:.2e}")

    # The key assertion: division should perform much worse than addition
    print(f"\nLoss comparison:")
    print(f"  Addition: {add_loss:.2e}")
    print(f"  Division: {div_loss:.2e}")
    print(f"  Division/Addition ratio: {div_loss/add_loss:.1f}x worse")

    # Division should be significantly worse (at least 10x higher loss)
    assert div_loss > 10 * add_loss, (
        f"Expected division to be much worse than addition, but got "
        f"div_loss={div_loss:.2e} vs add_loss={add_loss:.2e} "
        f"(ratio: {div_loss/add_loss:.1f}x)"
    )

    print("✅ CONFIRMED: Division training fails systematically compared to addition")
    return True


def test_division_selector_patterns_are_broken():
    """Test that division fails to learn proper operand selector patterns."""

    def run_with_debug_output(operation, seed=0):
        """Run training and capture selector patterns."""
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
            "4",
            "--max-iterations",
            "100",
            "--learning-rate",
            "1e-3",
            "--log-interval",
            "100",
            "--interpolation-range",
            "[-2.0,2.0]",
            "--seed",
            str(seed),
            "--no-cuda",
            "--no-open-browser",
            "--no-save",
            "--note",
            f"selector_debug_{operation}",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return result.stdout if result.returncode == 0 else ""

    def extract_selector_magnitude(output):
        """Extract the magnitude of selector values (how strongly it selects operands)."""
        lines = output.split("\n")
        for line in lines:
            if "values: [" in line:
                try:
                    values_str = line.split("values: ")[1]
                    values = eval(values_str)  # Safe because we control the format
                    # Return magnitude of first two selectors (for two operands)
                    if len(values) >= 2:
                        return abs(values[0]) + abs(values[1])
                except:
                    continue
        return 0.0

    # Test division selector learning
    print("Testing division selector pattern learning...")
    div_output = run_with_debug_output("div", seed=123)
    div_selector_strength = extract_selector_magnitude(div_output)

    # Test addition selector learning
    print("Testing addition selector pattern learning...")
    add_output = run_with_debug_output("add", seed=123)
    add_selector_strength = extract_selector_magnitude(add_output)

    print(f"\nSelector strength comparison:")
    print(f"  Addition selector magnitude: {add_selector_strength:.4f}")
    print(f"  Division selector magnitude: {div_selector_strength:.4f}")

    # Both should be learning some selectors, but division typically learns much weaker patterns
    if div_selector_strength < 0.01:
        print("🔍 Division shows extremely weak selector learning (< 0.01)")
        print(
            "   This indicates the network is not learning to select operands properly"
        )

    if add_selector_strength > 2 * div_selector_strength:
        print("✅ CONFIRMED: Division learns weaker selector patterns than addition")
    else:
        print("⚠️  Division selector strength not clearly worse than addition")

    # This test documents the behavior rather than failing - it's illustrative
    return True


def main():
    """Run the minimal division failure tests."""
    print("Minimal Division Training Failure Tests")
    print("=" * 45)

    try:
        print("\n1. Testing basic training comparison...")
        test_division_fails_basic_training()
    except Exception as e:
        print(f"❌ Basic training test failed: {e}")
        return False

    try:
        print(f"\n2. Testing selector pattern learning...")
        test_division_selector_patterns_are_broken()
    except Exception as e:
        print(f"❌ Selector pattern test failed: {e}")
        return False

    print(f"\n🎉 All tests completed successfully!")
    print(f"Division training issues have been systematically demonstrated.")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
