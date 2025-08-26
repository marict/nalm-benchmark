#!/usr/bin/env python3
"""
Re-run failed experiments with 10k iterations instead of 5k.
"""

import argparse
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

# Failed experiments to re-run
FAILED_EXPERIMENTS = [
    (122, "add", [-2, 2], [[-6, -2], [2, 6]]),
    (122, "add", [0.1, 0.2], [0.2, 2]),
    (122, "add", [10, 20], [20, 40]),
    (122, "sub", [10, 20], [20, 40]),
    (122, "div", [-20, -10], [-40, -20]),
    (223, "mul", [-0.2, -0.1], [-2, -0.2]),
    (223, "mul", [10, 20], [20, 40]),
    (223, "add", [0.1, 0.2], [0.2, 2]),
    (42, "mul", [10, 20], [20, 40]),
    (42, "add", [0.1, 0.2], [0.2, 2]),
    (42, "sub", [-20, -10], [-40, -20]),
    (777, "add", [0.1, 0.2], [0.2, 2]),
    (777, "sub", [-20, -10], [-40, -20]),
    (1337, "mul", [10, 20], [20, 40]),
    (1337, "add", [0.1, 0.2], [0.2, 2]),
]


def run_single_test(seed, operation, interp_range, extrap_range, show_progress=False):
    """Run a single test with 10k iterations and return result."""

    # Base command with 10k iterations
    cmd = [
        sys.executable,
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
        "10000",  # Increased from 5000
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

        # Use Popen for real-time output
        process = subprocess.Popen(
            cmd,
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
                line_stripped = line.rstrip()
                if show_progress:
                    # Show training progress lines and important status messages
                    if (
                        re.match(r"^train \d+:", line_stripped)
                        or "Early stopping" in line_stripped
                        or "Early stopped" in line_stripped
                        or "finished:" in line_stripped
                    ):
                        print(line_stripped)
                # Store all output for analysis
                output_lines.append(line_stripped)

            # Wait for process to complete (increased timeout for 10k iterations)
            process.wait(timeout=240)  # 4 minutes timeout instead of 2
            duration = time.time() - start_time

        except subprocess.TimeoutExpired:
            process.kill()
            duration = 240
            raise

        grokked = False
        grok_step = None
        final_inter_loss = float("inf")
        nan_error = False

        # Check for NaN errors first
        for line in output_lines:
            if "nan" in line.lower() or "NaN" in line:
                nan_error = True
                break

        # Check for early stopping
        if not nan_error:
            for line in output_lines:
                if "Early stopping at step" in line:
                    grokked = True
                    try:
                        grok_step = int(line.split("step ")[1].split(":")[0])
                    except:
                        pass
                    break

        # Check final loss if no early stopping and no NaN error
        if not grokked and not nan_error:
            for line in reversed(output_lines):
                if "- loss_valid_inter:" in line:
                    try:
                        final_inter_loss = float(line.split(":")[1].strip())
                        if math.isnan(final_inter_loss):
                            nan_error = True
                        elif final_inter_loss < 1e-8:
                            grokked = True
                        break
                    except:
                        continue

        # Determine status
        if nan_error:
            status = "nan_error"
        elif grokked:
            status = "grokking"
        else:
            status = "not_grokking"

        return {
            "operation": operation,
            "seed": seed,
            "interp_range": interp_range,
            "extrap_range": extrap_range,
            "frozen_config": frozen_config,
            "grokked": grokked,
            "nan_error": nan_error,
            "status": status,
            "grok_step": grok_step,
            "duration": duration,
            "final_inter_loss": final_inter_loss,
            "max_iterations": 10000,
        }

    except subprocess.TimeoutExpired:
        return {
            "operation": operation,
            "seed": seed,
            "interp_range": interp_range,
            "extrap_range": extrap_range,
            "frozen_config": frozen_config,
            "grokked": False,
            "nan_error": False,
            "status": "not_grokking",
            "grok_step": None,
            "duration": 240,
            "final_inter_loss": float("inf"),
            "max_iterations": 10000,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Re-run failed experiments with 10k iterations"
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Show training progress lines",
    )
    parser.add_argument(
        "--experiment",
        type=int,
        help="Run only the specified experiment number (1-15)",
    )
    args = parser.parse_args()

    print("RE-RUNNING FAILED EXPERIMENTS WITH 10K ITERATIONS")
    print("=" * 70)
    print(f"Total failed experiments: {len(FAILED_EXPERIMENTS)}")

    if args.experiment:
        if 1 <= args.experiment <= len(FAILED_EXPERIMENTS):
            experiments_to_run = [FAILED_EXPERIMENTS[args.experiment - 1]]
            print(f"Running only experiment #{args.experiment}")
        else:
            print(
                f"Error: Experiment number must be between 1 and {len(FAILED_EXPERIMENTS)}"
            )
            return 1
    else:
        experiments_to_run = FAILED_EXPERIMENTS

    print()

    results = []

    for i, (seed, operation, interp_range, extrap_range) in enumerate(
        experiments_to_run, 1
    ):
        print(
            f"Running experiment {i}/{len(experiments_to_run)}: Seed {seed} {operation.upper()} {interp_range} -> {extrap_range}"
        )

        result = run_single_test(
            seed, operation, interp_range, extrap_range, args.show_progress
        )
        results.append(result)

        # Display result
        if result["status"] == "grokking":
            print(
                f"✅ SUCCESS - Grokked at step {result['grok_step']} ({result['duration']:.1f}s)"
            )
        elif result["status"] == "nan_error":
            print(f"💥 NaN ERROR - ({result['duration']:.1f}s)")
        else:
            print(
                f"❌ STILL FAILED - Final loss: {result['final_inter_loss']:.6f} ({result['duration']:.1f}s)"
            )
        print()

    # Save results
    timestamp = int(time.time())
    output_dir = Path("experiment_results")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"failed_experiments_10k_results_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "config": {
                    "max_iterations": 10000,
                    "original_max_iterations": 5000,
                },
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"Results saved to: {output_file}")

    # Summary
    grokked_count = sum(1 for r in results if r["status"] == "grokking")
    nan_count = sum(1 for r in results if r["status"] == "nan_error")
    still_failed_count = sum(1 for r in results if r["status"] == "not_grokking")

    print(f"\nSUMMARY:")
    print(f"Total experiments: {len(results)}")
    print(f"Now grokking (✅): {grokked_count}")
    print(f"NaN errors (💥): {nan_count}")
    print(f"Still failing (❌): {still_failed_count}")
    print(f"Success rate with 10k iterations: {grokked_count/len(results)*100:.1f}%")


if __name__ == "__main__":
    main()
