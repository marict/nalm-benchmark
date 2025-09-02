#!/usr/bin/env python3
"""
Investigate why MUL n20/p20 ranges have such high thresholds.
"""

import json
import subprocess
import sys
from pathlib import Path


def run_detailed_test(operation, interp_range, extrap_range, range_name):
    """Run detailed test with full output to investigate the issue."""

    cmd = [
        sys.executable,
        "experiments/single_layer_benchmark.py",
        "--layer-type",
        "DAG",
        "--dag-depth",
        "1",
        "--operation",
        operation,
        "--weight-perturbation",
        "1e-5",
        "--seed",
        "0",
        "--input-size",
        "2",
        "--batch-size",
        "128",
        "--max-iterations",
        "1",
        "--interpolation-range",
        str(interp_range),
        "--extrapolation-range",
        str(extrap_range),
        "--no-cuda",
        "--freeze-input-norm",
        "--freeze-O-mul",
        "--freeze-G-log",
        "--unfreeze-eval",
        "--disable-early-stopping",
        "--disable-sounds",
        "--disable-logging",
        "--no-open-browser",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return result.stdout, result.stderr

    except Exception as e:
        return None, str(e)


def analyze_output(output, range_name):
    """Analyze the output for numerical issues."""
    print(f"\n{'='*60}")
    print(f"ANALYSIS FOR MUL {range_name.upper()} RANGE")
    print(f"{'='*60}")

    lines = output.split("\n")

    # Find key information
    for i, line in enumerate(lines):
        if "interpolation_range:" in line:
            print(f"Interpolation range: {line.strip()}")
        elif "extrapolation_range:" in line:
            print(f"Extrapolation range: {line.strip()}")
        elif "min|x| range:" in line:
            print(f"Input magnitude range: {line.strip()}")
        elif "Sample statistics (HARDENED eval state):" in line:
            # Print next 15 lines for sample analysis
            print(f"\nHARDENED EVAL SAMPLE:")
            for j in range(15):
                if i + j < len(lines):
                    print(f"  {lines[i+j]}")
        elif "loss_valid_inter:" in line:
            print(f"\nInterpolation loss: {line.strip()}")
        elif "loss_valid_extra:" in line:
            print(f"Extrapolation loss: {line.strip()}")
        elif "computed_value:" in line and "Step 0:" in lines[i - 1]:
            print(f"Computed output: {line.strip()}")
        elif "target=" in line:
            print(f"Target: {line.strip()}")


def main():
    print("INVESTIGATING MUL THRESHOLD ANOMALIES")
    print("=" * 50)

    # Test problematic ranges
    problem_ranges = [
        ([-20, -10], [-40, -20], "n20"),  # negative large
        ([10, 20], [20, 40], "p20"),  # positive large
    ]

    # Also test a good range for comparison
    good_ranges = [
        ([-2, 2], [[-6, -2], [2, 6]], "sym"),  # symmetric (good)
        ([0.1, 0.2], [0.2, 2], "p01"),  # small positive (very good)
    ]

    print("Testing problematic ranges:")
    for interp_range, extrap_range, range_name in problem_ranges:
        output, error = run_detailed_test("mul", interp_range, extrap_range, range_name)
        if output:
            analyze_output(output, range_name)
        else:
            print(f"ERROR for {range_name}: {error}")

    print("\n" + "=" * 60)
    print("COMPARISON WITH GOOD RANGES:")
    print("=" * 60)

    for interp_range, extrap_range, range_name in good_ranges:
        output, error = run_detailed_test("mul", interp_range, extrap_range, range_name)
        if output:
            analyze_output(output, range_name)
        else:
            print(f"ERROR for {range_name}: {error}")


if __name__ == "__main__":
    main()
