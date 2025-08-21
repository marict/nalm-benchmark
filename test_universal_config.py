#!/usr/bin/env python3
"""
Test if a combined config (tighter_clip + higher_lr) can grok all 3 operations.
Based on hyperparameter sensitivity results showing:
- ADD: higher_lr grokked
- MUL: tighter_clip grokked
- SUB: higher_lr grokked
"""

import subprocess
import sys
import time


def run_operation_test(operation: str, config_name: str, cmd_args: list) -> dict:
    """Run an operation test with specified configuration."""

    base_cmd = [
        sys.executable,
        "experiments/single_layer_benchmark.py",
        "--layer-type",
        "DAG",
        "--operation",
        operation,
        "--input-size",
        "2",
        "--max-iterations",
        "5000",
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
        f"{operation}_universal_{config_name}",
    ]

    cmd = base_cmd + cmd_args

    try:
        print(f"    Testing {operation.upper()}... ", end="", flush=True)
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
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
            print(f"✅ GROKKED! Error: {final_error:.2e} ({duration:.1f}s)")
        else:
            print(f"❌ No grok. Error: {final_error:.2e} ({duration:.1f}s)")

        return {
            "success": True,
            "final_error": final_error,
            "grokked": grokked,
            "duration": duration,
        }

    except subprocess.TimeoutExpired:
        print(f"❌ TIMEOUT (120s)")
        return {"success": False, "error": "timeout", "duration": 120.0}


def main():
    """Test universal grokking configuration."""
    print("Universal Grokking Configuration Test")
    print("=" * 40)
    print("Testing combined tighter_clip + higher_lr config on add/mul/sub")

    # Combined configuration: tighter_clip + higher_lr
    universal_config = [
        "--batch-size",
        "128",
        "--learning-rate",
        "2e-3",  # higher_lr
        "--lr-cosine",
        "--lr-min",
        "2e-4",
        "--clip-grad-norm",
        "0.5",  # tighter_clip
    ]

    operations = ["add", "mul", "sub"]
    results = {}

    print(f"\nTesting universal config on {len(operations)} operations:\n")

    for op in operations:
        result = run_operation_test(op, "universal", universal_config)
        results[op] = result

    # Analysis
    print(f"\n📊 RESULTS SUMMARY:")
    print(f"=" * 40)

    successful_ops = [op for op, res in results.items() if res.get("success", False)]
    grokked_ops = [op for op, res in results.items() if res.get("grokked", False)]

    if len(grokked_ops) == 3:
        print(f"🎉 UNIVERSAL SUCCESS! All 3 operations grokked:")
        for op in grokked_ops:
            res = results[op]
            print(
                f"   {op.upper()}: error={res['final_error']:.2e}, time={res['duration']:.1f}s"
            )

        print(f"\n🏆 Universal config found:")
        print(f"   --learning-rate 2e-3")
        print(f"   --clip-grad-norm 0.5")
        print(f"   --lr-cosine --lr-min 2e-4")
        print(f"   --batch-size 128")

        return True

    elif len(grokked_ops) >= 2:
        print(f"🔶 PARTIAL SUCCESS! {len(grokked_ops)}/3 operations grokked:")
        for op in grokked_ops:
            res = results[op]
            print(f"   ✅ {op.upper()}: error={res['final_error']:.2e}")

        failed_ops = [op for op in operations if op not in grokked_ops]
        for op in failed_ops:
            if op in successful_ops:
                res = results[op]
                print(f"   ❌ {op.upper()}: error={res['final_error']:.2e} (no grok)")
            else:
                print(f"   ❌ {op.upper()}: failed to run")

    else:
        print(f"❌ NO UNIVERSAL GROKKING")
        print(f"   Only {len(grokked_ops)}/3 operations grokked")

        # Show all results
        for op in operations:
            if op in successful_ops:
                res = results[op]
                status = "GROKKED" if res.get("grokked", False) else "no grok"
                print(f"   {op.upper()}: {res['final_error']:.2e} ({status})")
            else:
                print(f"   {op.upper()}: failed")

    return len(grokked_ops) == 3


if __name__ == "__main__":
    success = main()

    if success:
        print(f"\n🏆 SUCCESS: Found universal config that groks all operations!")
        print(f"   This config can be used as a reliable baseline for experiments.")
    else:
        print(f"\n🔍 No universal config found.")
        print(f"   Operations may need individual hyperparameter tuning.")

    sys.exit(0 if success else 1)
