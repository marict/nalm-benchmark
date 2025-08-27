#!/usr/bin/env python3
"""
Test script to debug restart logic issues.
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path


def run_experiment_with_detailed_logging(
    seed, operation, interp_range, extrap_range, restart_iter, max_iterations
):
    """Run a single experiment with detailed restart logging."""

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
        str(max_iterations),
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
        "--freeze-O-mul",
        "--restart-iter",
        str(restart_iter),
    ]

    print(f"Running: {' '.join(cmd)}")
    print("=" * 80)

    start_time = time.time()

    # Track key events
    events = {
        "restart_triggered": False,
        "early_stopping": False,
        "final_iteration": None,
        "restart_iteration": None,
        "early_stop_iteration": None,
        "duration": 0,
        "final_loss": None,
    }

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            universal_newlines=True,
            bufsize=1,
        )

        iteration_count = 0
        for line in process.stdout:
            line_stripped = line.rstrip()

            # Track training iterations
            if re.match(r"^train \d+:", line_stripped):
                iteration_match = re.match(r"^train (\d+):", line_stripped)
                if iteration_match:
                    iteration_count = int(iteration_match.group(1))
                    events["final_iteration"] = iteration_count
                print(line_stripped)

            # Track restart events
            elif "Restarting model at iteration" in line_stripped:
                events["restart_triggered"] = True
                restart_match = re.search(r"iteration (\d+)", line_stripped)
                if restart_match:
                    events["restart_iteration"] = int(restart_match.group(1))
                print(f"🔄 RESTART: {line_stripped}")

            # Track early stopping
            elif "Early stopping at step" in line_stripped:
                events["early_stopping"] = True
                early_stop_match = re.search(r"step (\d+)", line_stripped)
                if early_stop_match:
                    events["early_stop_iteration"] = int(early_stop_match.group(1))
                print(f"⭐ EARLY STOP: {line_stripped}")

            elif "Early stopped at step" in line_stripped:
                events["early_stopping"] = True
                early_stop_match = re.search(r"step (\d+)", line_stripped)
                if early_stop_match:
                    events["early_stop_iteration"] = int(early_stop_match.group(1))
                print(f"⭐ EARLY STOPPED: {line_stripped}")

            # Track final results
            elif "finished:" in line_stripped:
                print(f"🏁 FINISHED: {line_stripped}")

            # Track final loss
            elif "- loss_valid_inter:" in line_stripped:
                loss_match = re.search(
                    r"loss_valid_inter: ([\d\.e\-\+]+)", line_stripped
                )
                if loss_match:
                    events["final_loss"] = float(loss_match.group(1))
                print(f"📊 FINAL LOSS: {line_stripped}")

            # Show important configuration
            elif any(
                key in line_stripped
                for key in ["seed:", "operation:", "restart-iter:", "max_iterations:"]
            ):
                print(f"⚙️  CONFIG: {line_stripped}")

            # Show any errors or warnings
            elif any(
                word in line_stripped.lower()
                for word in ["error", "warning", "nan", "inf"]
            ):
                print(f"⚠️  ISSUE: {line_stripped}")

        process.wait(timeout=180)  # 3 minute timeout
        events["duration"] = time.time() - start_time

    except subprocess.TimeoutExpired:
        process.kill()
        events["duration"] = 180
        print("❌ TIMEOUT: Experiment exceeded 3 minutes")

    except Exception as e:
        print(f"❌ ERROR: {e}")
        events["duration"] = time.time() - start_time

    return events


def main():
    parser = argparse.ArgumentParser(description="Debug restart logic")
    parser.add_argument("--runs", type=int, default=3, help="Number of test runs")
    parser.add_argument("--seed", type=int, default=0, help="Seed to test")
    parser.add_argument("--operation", default="add", help="Operation to test")
    parser.add_argument(
        "--restart-iter", type=int, default=1500, help="Restart iteration"
    )
    parser.add_argument(
        "--max-iterations", type=int, default=3000, help="Max iterations"
    )
    args = parser.parse_args()

    # Test the problematic p01 range
    interp_range = [0.1, 0.2]
    extrap_range = [0.2, 2]

    print("RESTART LOGIC DEBUG TEST")
    print("=" * 80)
    print(f"Testing: seed={args.seed}, operation={args.operation}")
    print(f"Range: {interp_range} -> {extrap_range}")
    print(f"Restart at: {args.restart_iter}, Max: {args.max_iterations}")
    print(f"Running {args.runs} tests...")
    print()

    results = []

    for run in range(args.runs):
        print(f"\n🧪 RUN {run + 1}/{args.runs}")
        print("-" * 40)

        events = run_experiment_with_detailed_logging(
            args.seed,
            args.operation,
            interp_range,
            extrap_range,
            args.restart_iter,
            args.max_iterations,
        )

        results.append(events)

        print("\n📋 RUN SUMMARY:")
        print(f"  Duration: {events['duration']:.1f}s")
        print(f"  Final iteration: {events['final_iteration']}")
        print(f"  Restart triggered: {events['restart_triggered']}")
        if events["restart_triggered"]:
            print(f"  Restart at iteration: {events['restart_iteration']}")
        print(f"  Early stopping: {events['early_stopping']}")
        if events["early_stopping"]:
            print(f"  Early stop at iteration: {events['early_stop_iteration']}")
        print(f"  Final loss: {events['final_loss']}")

        # Analyze behavior
        success = events["early_stopping"]
        expected_restart = (
            events["final_iteration"] and events["final_iteration"] >= args.restart_iter
        )
        restart_happened = events["restart_triggered"]

        if expected_restart and not restart_happened:
            print("  ❌ ISSUE: Should have restarted but didn't")
        elif restart_happened and not expected_restart:
            print("  ❌ ISSUE: Restarted unexpectedly")
        elif restart_happened and not success:
            print("  ❌ ISSUE: Restarted but still failed")
        elif restart_happened and success:
            print("  ✅ SUCCESS: Restart led to early stopping")
        elif success and not restart_happened:
            print("  ✅ SUCCESS: Succeeded without restart")
        else:
            print("  ❌ FAILED: No restart and no success")

        print()

    # Overall analysis
    print("\n" + "=" * 80)
    print("OVERALL ANALYSIS")
    print("=" * 80)

    restart_count = sum(1 for r in results if r["restart_triggered"])
    success_count = sum(1 for r in results if r["early_stopping"])
    avg_duration = sum(r["duration"] for r in results) / len(results)

    print(f"Total runs: {len(results)}")
    print(
        f"Restarts triggered: {restart_count}/{len(results)} ({restart_count/len(results)*100:.1f}%)"
    )
    print(
        f"Successes (early stopping): {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)"
    )
    print(f"Average duration: {avg_duration:.1f}s")

    # Check for inconsistency
    if restart_count > 0 and restart_count < len(results):
        print(
            "\n❌ INCONSISTENCY DETECTED: Restart behavior varies between identical runs!"
        )
        print("This suggests a non-deterministic issue in the restart logic.")
    elif restart_count == len(results) and success_count < len(results):
        print("\n❌ RESTART INEFFECTIVENESS: All runs restart but some still fail.")
        print("This suggests the restart logic isn't working properly.")
    elif restart_count == len(results) and success_count == len(results):
        print("\n✅ CONSISTENT SUCCESS: All runs restart and succeed.")
    elif restart_count == 0 and success_count == len(results):
        print("\n✅ NO RESTART NEEDED: All runs succeed without restart.")
    else:
        print("\n❓ MIXED RESULTS: Need to investigate individual cases.")

    # Check duration patterns
    durations = [r["duration"] for r in results]
    if max(durations) - min(durations) > 20:  # More than 20 second variation
        print(f"\n⚠️  DURATION VARIATION: {min(durations):.1f}s - {max(durations):.1f}s")
        print("Large duration variation suggests different execution paths.")


if __name__ == "__main__":
    main()
