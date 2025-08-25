#!/usr/bin/env python3
"""
Comprehensive frozen selector test to generate percentage passing table.
"""

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Test configuration
TEST_SEEDS = [122, 223, 42, 777, 1337]
TEST_RANGES = [
    ([-2, 2], [[-6, -2], [2, 6]], "standard"),
    ([-2, -1], [-6, -2], "neg_moderate"),
    ([1, 2], [2, 6], "pos_moderate"),
    ([-1.2, -1.1], [-6.1, -1.2], "neg_narrow"),
    ([0.1, 0.2], [0.2, 2], "pos_small"),
    ([-0.2, -0.1], [-2, -0.2], "neg_small"),
    ([1.1, 1.2], [1.2, 6], "pos_narrow"),
    ([-20, -10], [-40, -20], "neg_large"),
    ([10, 20], [20, 40], "pos_large"),
]

OPERATIONS = ["mul", "add", "sub", "div"]


def run_single_test(operation, seed, interp_range, extrap_range):
    """Run a single test and return result."""

    # Base command with updated hyperparameters
    cmd = [
        "python",
        "experiments/single_layer_benchmark.py",
        "--layer-type",
        "DAG",
        "--no-open-browser",
        "--operation",
        operation,
        "--seed",
        str(seed),
        "--input-size",
        "2",
        "--batch-size",
        "512",
        "--max-iterations",
        "5000",
        "--learning-rate",
        "1e-3",
        "--interpolation-range",
        str(interp_range),
        "--extrapolation-range",
        str(extrap_range),
        "--no-cuda",
        "--log-interval",
        "100",
        "--clip-grad-norm",
        "0.01",
    ]

    # Add frozen selector arguments based on operation
    if operation in ["mul", "add"]:
        cmd.append("--freeze-O-mul")
        frozen_config = "freeze_O_mul=True"
    elif operation in ["sub", "div"]:
        cmd.append("--freeze-O-div")
        frozen_config = "freeze_O_div=True"

    try:
        start_time = time.time()
        # Keep --no-open-browser to prevent browser spam, tqdm still works
        cmd_display = cmd

        # Use Popen for real-time tqdm display
        process = subprocess.Popen(
            cmd_display,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            universal_newlines=True,
            bufsize=1,
        )

        output_lines = []
        try:
            # Read output line by line in real-time
            for line in process.stdout:
                print(line.rstrip())  # Print tqdm and other output in real-time
                output_lines.append(line.rstrip())

            # Wait for process to complete
            process.wait(timeout=120)
            duration = time.time() - start_time

        except subprocess.TimeoutExpired:
            process.kill()
            duration = 120
            raise

        grokked = False
        grok_step = None
        final_inter_loss = float("inf")

        # Check for early stopping
        for line in output_lines:
            if "Early stopping at step" in line:
                grokked = True
                try:
                    grok_step = int(line.split("step ")[1].split(":")[0])
                except:
                    pass
                break

        # Check final loss if no early stopping
        if not grokked:
            for line in reversed(output_lines):
                if "- loss_valid_inter:" in line:
                    try:
                        final_inter_loss = float(line.split(":")[1].strip())
                        if final_inter_loss < 1e-8:
                            grokked = True
                        break
                    except:
                        continue

        return {
            "operation": operation,
            "seed": seed,
            "interp_range": interp_range,
            "extrap_range": extrap_range,
            "frozen_config": frozen_config,
            "grokked": grokked,
            "grok_step": grok_step,
            "duration": duration,
            "final_inter_loss": final_inter_loss,
        }

    except subprocess.TimeoutExpired:
        return {
            "operation": operation,
            "seed": seed,
            "interp_range": interp_range,
            "extrap_range": extrap_range,
            "frozen_config": frozen_config,
            "grokked": False,
            "grok_step": None,
            "duration": 120,
            "final_inter_loss": float("inf"),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive frozen selector table test"
    )
    parser.add_argument(
        "--max-experiments",
        type=int,
        help="Limit total number of experiments (useful for debugging)",
    )
    args = parser.parse_args()

    results = []

    print("COMPREHENSIVE FROZEN SELECTOR TABLE GENERATION")
    print("=" * 70)
    print(f"Operations: {OPERATIONS}")
    print(f"Seeds: {TEST_SEEDS}")
    print(f"Ranges: {len(TEST_RANGES)} different interpolation/extrapolation pairs")

    total_experiments = len(OPERATIONS) * len(TEST_SEEDS) * len(TEST_RANGES)
    if args.max_experiments:
        total_experiments = min(total_experiments, args.max_experiments)
        print(f"Total experiments: {total_experiments} (LIMITED)")
    else:
        print(f"Total experiments: {total_experiments}")
    print()

    completed = 0

    for interp_range, extrap_range, range_name in TEST_RANGES:
        print(f"\n🔸 Testing range {range_name}: {interp_range} → {extrap_range}")

        for operation in OPERATIONS:
            print(f"  {operation.upper()}: ", end="")

            for seed in TEST_SEEDS:
                # Check if we've reached the experiment limit
                if args.max_experiments and completed >= args.max_experiments:
                    print(
                        f"\n⚠️ Reached experiment limit ({args.max_experiments}), stopping..."
                    )
                    break

                result = run_single_test(operation, seed, interp_range, extrap_range)
                results.append(result)
                completed += 1

                if result["grokked"]:
                    print("✅", end="")
                else:
                    print("❌", end="")

                print(f"({seed})", end=" ")

            # Early break from operations loop
            if args.max_experiments and completed >= args.max_experiments:
                break

            # Calculate success rate for this operation/range combo
            op_range_results = [
                r
                for r in results
                if r["operation"] == operation
                and str(r["interp_range"]) == str(interp_range)
            ]
            success_rate = (
                sum(1 for r in op_range_results if r["grokked"])
                / len(op_range_results)
                * 100
            )
            print(f"→ {success_rate:.0f}%")

        print(
            f"  Progress: {completed}/{total_experiments} ({completed/total_experiments*100:.1f}%)"
        )

        # Early break from ranges loop
        if args.max_experiments and completed >= args.max_experiments:
            break

    # Generate comprehensive table
    print(f"\n" + "=" * 70)
    print("RESULTS TABLE")
    print("=" * 70)

    # Create DataFrame for analysis
    data = []
    for interp_range, extrap_range, range_name in TEST_RANGES:
        for operation in OPERATIONS:
            op_range_results = [
                r
                for r in results
                if r["operation"] == operation
                and str(r["interp_range"]) == str(interp_range)
            ]

            grokked_count = sum(1 for r in op_range_results if r["grokked"])
            total_count = len(op_range_results)
            success_rate = grokked_count / total_count * 100 if total_count > 0 else 0

            # Get average grok step for successful runs
            grok_steps = [
                r["grok_step"] for r in op_range_results if r["grok_step"] is not None
            ]
            avg_grok_step = sum(grok_steps) / len(grok_steps) if grok_steps else None

            data.append(
                {
                    "Range": range_name,
                    "Interp": str(interp_range),
                    "Extrap": str(extrap_range),
                    "Operation": operation.upper(),
                    "Success_Rate": f"{success_rate:.0f}%",
                    "Count": f"{grokked_count}/{total_count}",
                    "Avg_Step": f"{avg_grok_step:.0f}" if avg_grok_step else "N/A",
                }
            )

    # Print table
    df = pd.DataFrame(data)

    # Pivot table for better display
    pivot_success = df.pivot(
        index=["Range", "Interp", "Extrap"], columns="Operation", values="Success_Rate"
    )
    pivot_steps = df.pivot(
        index=["Range", "Interp", "Extrap"], columns="Operation", values="Avg_Step"
    )

    print("\nSUCCESS RATES (% passing across seeds):")
    print(pivot_success.to_string())

    print(f"\nAVERAGE GROK STEPS (for successful runs):")
    print(pivot_steps.to_string())

    # Save results
    timestamp = int(time.time())
    output_dir = Path("experiment_results")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"comprehensive_table_results_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "config": {
                    "seeds": TEST_SEEDS,
                    "ranges": [(ir, er, name) for ir, er, name in TEST_RANGES],
                    "operations": OPERATIONS,
                },
                "results": results,
                "summary_table": data,
            },
            f,
            indent=2,
        )

    print(f"\nDetailed results saved to: {output_file}")

    # Overall summary
    total_grokked = sum(1 for r in results if r["grokked"])
    total_experiments = len(results)
    overall_success_rate = total_grokked / total_experiments * 100

    print(f"\nOVERALL SUMMARY:")
    print(f"Total experiments: {total_experiments}")
    print(f"Total successful: {total_grokked}")
    print(f"Overall success rate: {overall_success_rate:.1f}%")


if __name__ == "__main__":
    main()
