#!/usr/bin/env python3
"""
Test different DAG architectural flags for division training.
Maybe division benefits from different STE settings or normalization.
"""

import subprocess
import sys
import time


def test_division_with_flags(flags_desc: str, extra_args: list, seed: int = 0) -> dict:
    """Test division with specific architectural flags."""

    base_cmd = [
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
        "1500",  # Shorter for quick testing
        "--learning-rate",
        "1e-3",
        "--log-interval",
        "500",
        "--interpolation-range",
        "[-2.0,2.0]",
        "--extrapolation-range",
        "[[-6.0,-2.0],[2.0,6.0]]",
        "--seed",
        str(seed),
        "--no-cuda",
        "--no-open-browser",
        "--no-save",
        "--lr-cosine",
        "--lr-min",
        "1e-4",
        "--clip-grad-norm",
        "1.0",
        "--note",
        f"div_arch_test_{flags_desc}",
    ]

    cmd = base_cmd + extra_args

    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        duration = time.time() - start_time

        if result.returncode != 0:
            return {
                "success": False,
                "error": result.stderr[:200],  # Truncate error
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

        return {
            "success": True,
            "final_error": final_error,
            "grokked": grokked,
            "duration": duration,
        }

    except subprocess.TimeoutExpired:
        return {"success": False, "error": "timeout", "duration": 90.0}


def main():
    """Test different architectural configurations for division."""
    print("Division Architectural Flag Testing")
    print("=" * 40)

    # Test different combinations of architectural flags
    # Based on the DAG layer flags we saw in dag.py
    test_configs = [
        ("baseline", []),
        # Test STE variations
        (
            "ste_all",
            ["--use-ste-G", "--use-ste-sign", "--use-ste-O-mag", "--use-ste-O-sign"],
        ),
        ("ste_selective", ["--use-ste-G", "--use-ste-sign"]),
        ("no_ste", []),  # This is already baseline
        # Test normalization
        ("no_norm", ["--no-use-normalization"]),
        # Test temperature removal
        ("with_temps", ["--no-remove-temperatures"]),
        # Test domain mixing
        ("complex_mixing", ["--no-use-simple-domain-mixing"]),
        # Test unified selector
        ("unified_sel", ["--use-unified-selector"]),
        # Combined potentially helpful flags
        ("kitchen_sink", ["--use-ste-G", "--use-ste-sign", "--use-unified-selector"]),
    ]

    results = []

    for config_name, flags in test_configs:
        print(f"\n🔧 Testing {config_name}")
        print(f"   Flags: {' '.join(flags) if flags else 'None'}")
        print("   ", end="", flush=True)

        result = test_division_with_flags(config_name, flags, seed=42)
        results.append((config_name, flags, result))

        if result["success"]:
            error = result["final_error"]
            grok_status = "✅ GROKKED" if result["grokked"] else "❌ No grok"
            print(
                f"Final error: {error:.2e}, {grok_status} ({result['duration']:.1f}s)"
            )
        else:
            print(f"❌ FAILED: {result.get('error', 'unknown')[:50]}...")

    # Analysis
    print(f"\n📊 ANALYSIS:")
    print(f"=" * 40)

    successful_results = [
        (name, flags, res) for name, flags, res in results if res["success"]
    ]
    if successful_results:
        # Sort by final error (lower is better)
        successful_results.sort(key=lambda x: x[2]["final_error"] or float("inf"))

        print(f"Configuration rankings (by final error):")
        for i, (name, flags, res) in enumerate(successful_results):
            error = res["final_error"]
            grok = "🎉 GROK" if res["grokked"] else "No grok"
            print(f"  {i+1:2d}. {name:15s}: {error:.2e} ({grok})")

        # Check for grokking
        grokked_configs = [
            (name, flags, res)
            for name, flags, res in successful_results
            if res["grokked"]
        ]
        if grokked_configs:
            print(f"\n🎉 BREAKTHROUGH! These configs achieved grokking:")
            for name, flags, res in grokked_configs:
                print(f"     {name}: {' '.join(flags) if flags else 'baseline'}")
                print(f"       Final error: {res['final_error']:.2e}")
        else:
            print(f"\n🔍 No configuration achieved grokking.")
            best_name, best_flags, best_res = successful_results[0]
            improvement = best_res["final_error"]
            print(f"   Best configuration: {best_name}")
            print(f"   Best error: {improvement:.2e}")

            # Compare to baseline
            baseline_res = next(
                (res for name, flags, res in results if name == "baseline"), None
            )
            if baseline_res and baseline_res["success"]:
                baseline_error = baseline_res["final_error"]
                if improvement < baseline_error:
                    ratio = baseline_error / improvement
                    print(f"   Improvement over baseline: {ratio:.1f}x better")
                else:
                    print(f"   No improvement over baseline")

    else:
        print(f"❌ All configurations failed!")

    # Return whether any config achieved grokking
    return any(res.get("grokked", False) for _, _, res in results)


if __name__ == "__main__":
    success = main()
    if success:
        print(
            f"\n🎉 SUCCESS: Found architectural configuration that enables division grokking!"
        )
    else:
        print(
            f"\n🔬 No grokking found, but may have found improvements to investigate further."
        )

    sys.exit(0 if success else 1)
