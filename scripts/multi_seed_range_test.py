#!/usr/bin/env python3
"""
Multi-seed, multi-range frozen selector test with reduced scope for faster results.
"""

import json
import subprocess
import time

# Test configuration - reduced for faster results
TEST_SEEDS = [122, 223, 42]  # 3 seeds
TEST_RANGES = [
    ([-2, 2], [[-6, -2], [2, 6]]),  # Standard good range
    ([1, 2], [2, 6]),  # Positive range
    ([-0.2, -0.1], [-2, -0.2]),  # Challenging small range
]


def modify_dag_defaults(freeze_div=False, freeze_mul=False):
    """Modify DAG defaults by editing the file directly."""
    dag_file = "/Users/paul_curry/ai2/nalm-benchmark/stable_nalu/layer/dag.py"

    with open(dag_file, "r") as f:
        content = f.read()

    if freeze_div and not freeze_mul:
        content = content.replace(
            "freeze_O_selectors_div: bool = False,",
            "freeze_O_selectors_div: bool = True,",
        ).replace(
            "freeze_O_selector_mul: bool = True,",
            "freeze_O_selector_mul: bool = False,",
        )
    elif freeze_mul and not freeze_div:
        content = content.replace(
            "freeze_O_selectors_div: bool = True,",
            "freeze_O_selectors_div: bool = False,",
        ).replace(
            "freeze_O_selector_mul: bool = False,",
            "freeze_O_selector_mul: bool = True,",
        )

    with open(dag_file, "w") as f:
        f.write(content)


def run_experiment(operation, seed, interp_range, extrap_range):
    """Run a single experiment."""
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
        "2000",
        "--learning-rate",
        "1e-2",
        "--interpolation-range",
        str(interp_range),
        "--extrapolation-range",
        str(extrap_range),
        "--no-cuda",
        "--log-interval",
        "500",
    ]

    print(f"  {operation} seed {seed} {interp_range}", end="...")

    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        duration = time.time() - start_time

        output_lines = result.stdout.split("\n") if result.stdout else []

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

        if grokked:
            print(f" ✅ @{grok_step}")
        else:
            print(f" ❌ loss:{final_inter_loss:.1e}")

        return {
            "operation": operation,
            "seed": seed,
            "interp_range": interp_range,
            "extrap_range": extrap_range,
            "grokked": grokked,
            "grok_step": grok_step,
            "final_inter_loss": final_inter_loss,
        }

    except subprocess.TimeoutExpired:
        print(f" ⏰ timeout")
        return {
            "operation": operation,
            "seed": seed,
            "interp_range": interp_range,
            "extrap_range": extrap_range,
            "grokked": False,
            "grok_step": None,
            "final_inter_loss": float("inf"),
        }


def main():
    results = []

    print("MULTI-SEED MULTI-RANGE FROZEN SELECTOR TEST")
    print("=" * 60)
    print(f"Seeds: {TEST_SEEDS}")
    print(f"Ranges: {len(TEST_RANGES)} ranges")
    print(f"Total experiments: {len(TEST_SEEDS) * len(TEST_RANGES) * 4}")

    # Group 1: mul/add with freeze_O_selector_mul
    print(f"\n🔸 MUL/ADD with freeze_O_selector_mul=True")
    modify_dag_defaults(freeze_div=False, freeze_mul=True)

    for operation in ["mul", "add"]:
        print(f"\n{operation.upper()}:")
        for seed in TEST_SEEDS:
            for interp_range, extrap_range in TEST_RANGES:
                result = run_experiment(operation, seed, interp_range, extrap_range)
                results.append(result)

    # Group 2: sub/div with freeze_O_selectors_div
    print(f"\n🔸 SUB/DIV with freeze_O_selectors_div=True")
    modify_dag_defaults(freeze_div=True, freeze_mul=False)

    for operation in ["sub", "div"]:
        print(f"\n{operation.upper()}:")
        for seed in TEST_SEEDS:
            for interp_range, extrap_range in TEST_RANGES:
                result = run_experiment(operation, seed, interp_range, extrap_range)
                results.append(result)

    # Analyze results
    print(f"\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    by_operation = {}
    by_range = {}

    for result in results:
        op = result["operation"]
        range_key = str(result["interp_range"])

        if op not in by_operation:
            by_operation[op] = []
        by_operation[op].append(result)

        if range_key not in by_range:
            by_range[range_key] = []
        by_range[range_key].append(result)

    # Per-operation results
    print("Per-operation success rates:")
    for op, op_results in by_operation.items():
        grokked_count = sum(1 for r in op_results if r["grokked"])
        total_count = len(op_results)
        success_rate = grokked_count / total_count if total_count > 0 else 0

        avg_grok_step = None
        grok_steps = [r["grok_step"] for r in op_results if r["grok_step"] is not None]
        if grok_steps:
            avg_grok_step = sum(grok_steps) / len(grok_steps)

        print(
            f"  {op.upper()}: {success_rate:.1%} ({grokked_count}/{total_count})"
            + (f", avg step: {avg_grok_step:.0f}" if avg_grok_step else "")
        )

    # Per-range results
    print("\nPer-range success rates:")
    for range_key, range_results in by_range.items():
        grokked_count = sum(1 for r in range_results if r["grokked"])
        total_count = len(range_results)
        success_rate = grokked_count / total_count if total_count > 0 else 0
        print(f"  {range_key}: {success_rate:.1%} ({grokked_count}/{total_count})")

    # Overall
    total_grokked = sum(1 for r in results if r["grokked"])
    total_experiments = len(results)
    overall_success_rate = (
        total_grokked / total_experiments if total_experiments > 0 else 0
    )

    print(
        f"\nOVERALL: {overall_success_rate:.1%} ({total_grokked}/{total_experiments})"
    )

    return results


if __name__ == "__main__":
    main()
