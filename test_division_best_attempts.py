#!/usr/bin/env python3
"""
Test the most promising hyperparameter combinations for division training.
Based on preliminary results from individual experiments.
"""

import subprocess
import sys
import time


def run_division_experiment(config_name: str, cmd_args: list) -> dict:
    """Run a division experiment with specified configuration."""

    base_cmd = [
        sys.executable,
        "experiments/single_layer_benchmark.py",
        "--layer-type",
        "DAG",
        "--operation",
        "div",
        "--input-size",
        "2",
        "--max-iterations",
        "3000",
        "--log-interval",
        "1000",
        "--interpolation-range",
        "[-2.0,2.0]",
        "--extrapolation-range",
        "[[-6.0,-2.0],[2.0,6.0]]",
        "--seed",
        "42",  # Fixed seed for comparison
        "--no-cuda",
        "--no-open-browser",
        "--no-save",
        "--note",
        f"div_best_attempt_{config_name}",
    ]

    cmd = base_cmd + cmd_args

    try:
        print(f"    Running: {config_name}... ", end="", flush=True)
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        duration = time.time() - start_time

        if result.returncode != 0:
            print(f"❌ FAILED ({duration:.1f}s)")
            return {
                "success": False,
                "error": result.stderr[:100],
                "duration": duration,
            }

        # Parse results
        output = result.stdout
        final_error = None
        grokked = "Early stopping" in output

        # Extract final interpolation error
        lines = output.split("\n")
        for line in reversed(lines):
            if "inter:" in line and "train" in line:
                try:
                    inter_part = line.split("inter: ")[1].split(",")[0]
                    final_error = float(inter_part)
                    break
                except (IndexError, ValueError):
                    continue

        if grokked:
            print(f"🎉 GROKKED! Error: {final_error:.2e} ({duration:.1f}s)")
        else:
            print(f"❌ No grok. Error: {final_error:.2e} ({duration:.1f}s)")

        return {
            "success": True,
            "final_error": final_error,
            "grokked": grokked,
            "duration": duration,
        }

    except subprocess.TimeoutExpired:
        print(f"❌ TIMEOUT (180s)")
        return {"success": False, "error": "timeout", "duration": 180.0}


def main():
    """Test the most promising configurations for division."""
    print("Division Best Attempts Test")
    print("=" * 30)
    print("Testing most promising hyperparameter combinations...")

    # Based on what typically helps with difficult optimization problems
    promising_configs = [
        # Baseline for comparison
        (
            "baseline",
            [
                "--batch-size",
                "128",
                "--learning-rate",
                "1e-3",
                "--lr-cosine",
                "--lr-min",
                "1e-4",
                "--clip-grad-norm",
                "1.0",
            ],
        ),
        # Lower learning rate + larger batch (more stable gradients)
        (
            "stable_training",
            [
                "--batch-size",
                "512",
                "--learning-rate",
                "5e-4",
                "--lr-cosine",
                "--lr-min",
                "1e-5",
                "--clip-grad-norm",
                "0.5",
            ],
        ),
        # SGD with momentum (sometimes better for discrete problems)
        (
            "sgd_momentum",
            [
                "--batch-size",
                "256",
                "--learning-rate",
                "1e-2",
                "--optimizer",
                "sgd",
                "--momentum",
                "0.9",
                "--clip-grad-norm",
                "1.0",
            ],
        ),
        # Constant LR (no scheduling)
        (
            "constant_lr",
            [
                "--batch-size",
                "128",
                "--learning-rate",
                "1e-3",
                "--clip-grad-norm",
                "1.0",
            ],
        ),
        # Higher LR with tighter clipping
        (
            "aggressive",
            [
                "--batch-size",
                "128",
                "--learning-rate",
                "2e-3",
                "--lr-cosine",
                "--lr-min",
                "2e-4",
                "--clip-grad-norm",
                "0.3",
            ],
        ),
        # Very long training with patient LR
        (
            "patient_training",
            [
                "--batch-size",
                "256",
                "--learning-rate",
                "8e-4",
                "--lr-cosine",
                "--lr-min",
                "5e-6",
                "--clip-grad-norm",
                "0.8",
                "--max-iterations",
                "8000",  # Override base max-iterations
            ],
        ),
    ]

    print(f"\nTesting {len(promising_configs)} configurations:\n")

    results = []

    for config_name, args in promising_configs:
        result = run_division_experiment(config_name, args)
        results.append((config_name, result))

    # Analysis
    print(f"\n📊 RESULTS SUMMARY:")
    print(f"=" * 30)

    successful_results = [(name, res) for name, res in results if res["success"]]
    if successful_results:
        # Sort by final error
        successful_results.sort(key=lambda x: x[1]["final_error"] or float("inf"))

        # Check for any grokking
        grokked_configs = [
            (name, res) for name, res in successful_results if res["grokked"]
        ]

        if grokked_configs:
            print(f"🎉 BREAKTHROUGH! Division grokked with:")
            for name, res in grokked_configs:
                print(
                    f"   {name}: error={res['final_error']:.2e}, time={res['duration']:.1f}s"
                )

            return True  # Success!

        else:
            print(f"❌ No grokking achieved, but ranking by final error:")
            for i, (name, res) in enumerate(successful_results[:3]):  # Top 3
                error = res["final_error"]
                print(f"   {i+1}. {name}: {error:.2e}")

            # Compare best to baseline
            baseline_res = next(
                (res for name, res in results if name == "baseline"), None
            )
            best_res = successful_results[0][1]

            if baseline_res and baseline_res["success"]:
                baseline_error = baseline_res["final_error"]
                best_error = best_res["final_error"]

                if best_error < baseline_error:
                    improvement = baseline_error / best_error
                    print(f"   Best config is {improvement:.1f}x better than baseline!")
                else:
                    print(f"   No significant improvement over baseline.")

    else:
        print(f"❌ All configurations failed!")

    return False


if __name__ == "__main__":
    success = main()

    if success:
        print(
            f"\n🏆 SUCCESS: Found hyperparameter configuration that enables division grokking!"
        )
    else:
        print(
            f"\n🔬 No grokking achieved. Division remains a challenging training problem."
        )
        print(f"   The issue may be architectural rather than hyperparameter-related.")

    sys.exit(0 if success else 1)
