#!/usr/bin/env python3
"""
Test hyperparameter sensitivity for add/mul/sub operations.
Ensures that our division-focused hyperparameter changes don't break working operations.
"""

import subprocess
import sys
import time


def test_operation_with_params(
    operation: str, params: dict, max_iterations: int = 1000
) -> dict:
    """Test an operation with specific hyperparameters."""

    # Build command
    cmd = [
        sys.executable,
        "experiments/single_layer_benchmark.py",
        "--layer-type",
        "DAG",
        "--operation",
        operation,
        "--input-size",
        "2",
        "--max-iterations",
        str(max_iterations),
        "--log-interval",
        str(max_iterations),
        "--interpolation-range",
        "[-2.0,2.0]",
        "--extrapolation-range",
        "[[-6.0,-2.0],[2.0,6.0]]",
        "--seed",
        "42",  # Fixed seed for consistency
        "--no-cuda",
        "--no-open-browser",
        "--no-save",
        "--note",
        f"hyperparam_test_{operation}",
    ]

    # Add hyperparameter-specific arguments
    for key, value in params.items():
        if key == "batch_size":
            cmd.extend(["--batch-size", str(value)])
        elif key == "learning_rate":
            cmd.extend(["--learning-rate", str(value)])
        elif key == "lr_cosine":
            if value:
                cmd.extend(["--lr-cosine", "--lr-min", "1e-4"])
        elif key == "clip_grad_norm":
            cmd.extend(["--clip-grad-norm", str(value)])

    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        duration = time.time() - start_time

        if result.returncode != 0:
            return {"success": False, "error": result.stderr, "duration": duration}

        # Parse final interpolation error
        final_error = None
        lines = result.stdout.split("\n")
        for line in reversed(lines):
            if "inter:" in line and "train" in line:
                try:
                    inter_part = line.split("inter: ")[1].split(",")[0]
                    final_error = float(inter_part)
                    break
                except (IndexError, ValueError):
                    continue

        # Check for early stopping (grokking)
        grokked = "Early stopping" in result.stdout

        return {
            "success": True,
            "final_error": final_error,
            "grokked": grokked,
            "duration": duration,
            "output": result.stdout,
        }

    except subprocess.TimeoutExpired:
        return {"success": False, "error": "timeout", "duration": 180.0}


def main():
    """Test different hyperparameter configurations."""
    print("Hyperparameter Sensitivity Test for Add/Mul/Sub")
    print("=" * 50)

    # Baseline configuration (what currently works)
    baseline = {
        "batch_size": 128,
        "learning_rate": 1e-3,
        "lr_cosine": True,
        "clip_grad_norm": 1.0,
    }

    # Alternative configurations to test for division
    configurations = {
        "baseline": baseline,
        "lower_lr": {**baseline, "learning_rate": 5e-4},
        "higher_lr": {**baseline, "learning_rate": 2e-3},
        "larger_batch": {**baseline, "batch_size": 512},
        "tighter_clip": {**baseline, "clip_grad_norm": 0.5},
        "constant_lr": {**baseline, "lr_cosine": False},
    }

    # Test operations that should work
    operations = ["add", "mul", "sub"]

    results = {}

    for op in operations:
        print(f"\n🧮 Testing {op.upper()} operation:")
        op_results = {}

        for config_name, config in configurations.items():
            print(f"  Testing {config_name}... ", end="", flush=True)

            result = test_operation_with_params(op, config, max_iterations=800)
            op_results[config_name] = result

            if result["success"]:
                error = result["final_error"]
                grok = "✅ GROKKED" if result["grokked"] else "🟡 No grok"
                print(f"Final error: {error:.2e}, {grok} ({result['duration']:.1f}s)")
            else:
                print(f"❌ FAILED: {result.get('error', 'unknown')}")

        results[op] = op_results

    # Summary analysis
    print(f"\n📊 SUMMARY:")
    print(f"=" * 50)

    for op in operations:
        print(f"\n{op.upper()} Operation Results:")

        baseline_result = results[op]["baseline"]
        baseline_error = baseline_result.get("final_error", float("inf"))
        baseline_grokked = baseline_result.get("grokked", False)

        print(f"  Baseline: error={baseline_error:.2e}, grokked={baseline_grokked}")

        # Check if other configs maintain similar performance
        for config_name, result in results[op].items():
            if config_name == "baseline":
                continue

            if not result["success"]:
                print(f"  {config_name}: ❌ FAILED")
                continue

            error = result.get("final_error", float("inf"))
            grokked = result.get("grokked", False)

            # Compare to baseline
            if baseline_result["success"]:
                if error < 2 * baseline_error:  # Within 2x of baseline
                    status = "✅ GOOD"
                elif error < 10 * baseline_error:  # Within 10x
                    status = "🟡 OK"
                else:
                    status = "❌ WORSE"
            else:
                status = "✅ BETTER" if result["success"] else "❌ FAILED"

            print(f"  {config_name}: error={error:.2e}, grokked={grokked} {status}")

    # Recommend configurations that work well across operations
    print(f"\n🎯 RECOMMENDATIONS:")

    # Find configs that work well for all operations
    good_configs = []
    for config_name in configurations.keys():
        if config_name == "baseline":
            continue

        works_for_all = True
        for op in operations:
            result = results[op][config_name]
            baseline = results[op]["baseline"]

            if not result["success"]:
                works_for_all = False
                break

            # Check if performance is reasonable compared to baseline
            if baseline["success"]:
                error_ratio = result.get("final_error", float("inf")) / baseline.get(
                    "final_error", 1.0
                )
                if error_ratio > 10:  # Much worse than baseline
                    works_for_all = False
                    break

        if works_for_all:
            good_configs.append(config_name)

    if good_configs:
        print(f"✅ These configs work well for add/mul/sub: {', '.join(good_configs)}")
        print(
            f"   These are safe to try for division without breaking working operations."
        )
    else:
        print(
            f"⚠️  No alternative configs clearly better than baseline for all operations."
        )
        print(f"   Stick with baseline for add/mul/sub when testing division fixes.")


if __name__ == "__main__":
    main()
