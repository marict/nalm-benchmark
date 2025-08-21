#!/usr/bin/env python3
"""
Test different optimizers for division training.
Maybe division needs SGD instead of Adam for better convergence.
"""

import subprocess
import sys
import time


def test_division_with_optimizer(optimizer: str, lr: float, seed: int = 0) -> dict:
    """Test division with specific optimizer."""

    cmd = [
        sys.executable,
        "experiments/single_layer_benchmark.py",
        "--layer-type",
        "DAG",
        "--operation",
        "div",
        "--input-size",
        "2",
        "--batch-size",
        "128",
        "--max-iterations",
        "2000",
        "--learning-rate",
        str(lr),
        "--log-interval",
        "500",
        "--interpolation-range",
        "[-2.0,2.0]",
        "--extrapolation-range",
        "[[-6.0,-2.0],[2.0,6.0]]",
        "--optimizer",
        optimizer,
        "--seed",
        str(seed),
        "--no-cuda",
        "--no-open-browser",
        "--no-save",
        "--note",
        f"div_optimizer_{optimizer}_test",
    ]

    # Add optimizer-specific parameters
    if optimizer == "sgd":
        cmd.extend(["--momentum", "0.9"])  # SGD with momentum

    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        duration = time.time() - start_time

        if result.returncode != 0:
            return {"success": False, "error": result.stderr, "duration": duration}

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

        return {
            "success": True,
            "final_error": final_error,
            "grokked": grokked,
            "duration": duration,
        }

    except subprocess.TimeoutExpired:
        return {"success": False, "error": "timeout", "duration": 120.0}


def main():
    """Test different optimizers for division."""
    print("Division Optimizer Comparison")
    print("=" * 35)

    # Test configurations
    configs = [
        ("adam", 1e-3),
        ("adam", 5e-4),
        ("adam", 2e-3),
        ("sgd", 1e-2),  # SGD usually needs higher LR
        ("sgd", 5e-3),
        ("sgd", 2e-2),
    ]

    results = []

    for optimizer, lr in configs:
        print(f"\n🔧 Testing {optimizer.upper()} with LR={lr:.0e}")
        print("  ", end="", flush=True)

        result = test_division_with_optimizer(optimizer, lr, seed=42)
        results.append((optimizer, lr, result))

        if result["success"]:
            error = result["final_error"]
            grok_status = "✅ GROKKED" if result["grokked"] else "❌ No grok"
            print(
                f"Final error: {error:.2e}, {grok_status} ({result['duration']:.1f}s)"
            )
        else:
            print(f"❌ FAILED: {result.get('error', 'unknown')}")

    # Find best performing configuration
    print(f"\n📊 SUMMARY:")
    print(f"=" * 35)

    successful_results = [(opt, lr, res) for opt, lr, res in results if res["success"]]
    if successful_results:
        # Sort by final error (lower is better)
        successful_results.sort(key=lambda x: x[2]["final_error"] or float("inf"))

        print(f"Best performing configurations:")
        for i, (opt, lr, res) in enumerate(successful_results[:3]):  # Top 3
            error = res["final_error"]
            grok = res["grokked"]
            print(
                f"  {i+1}. {opt.upper()} LR={lr:.0e}: error={error:.2e}, grokked={grok}"
            )

        # Check if any grokked
        grokked_configs = [
            (opt, lr, res) for opt, lr, res in successful_results if res["grokked"]
        ]
        if grokked_configs:
            print(f"\n🎉 Grokking achieved with:")
            for opt, lr, res in grokked_configs:
                print(
                    f"     {opt.upper()} LR={lr:.0e} (error: {res['final_error']:.2e})"
                )
        else:
            print(f"\n🔍 No configuration achieved grokking for division.")
            print(
                f"   Best error achieved: {successful_results[0][2]['final_error']:.2e}"
            )

    else:
        print(f"❌ All optimizer configurations failed!")

    return len([res for _, _, res in results if res.get("grokked", False)]) > 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
