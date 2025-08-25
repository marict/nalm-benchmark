#!/usr/bin/env python3
"""
Quick test of frozen selector performance on a few key ranges.
"""

import subprocess
import time

# Quick test configuration
TEST_SEEDS = [122]
QUICK_RANGES = [
    ([-2, 2], [[-6, -2], [2, 6]]),  # Good range
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


def run_quick_test(operation, seed, interp_range, extrap_range):
    """Run a single quick test."""
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

    print(f"Testing {operation} seed {seed} range {interp_range}...", end=" ")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        output_lines = result.stdout.split("\n") if result.stdout else []

        grokked = False
        grok_step = None

        # Check for early stopping
        for line in output_lines:
            if "Early stopping at step" in line:
                grokked = True
                try:
                    grok_step = int(line.split("step ")[1].split(":")[0])
                except:
                    pass
                break

        if grokked:
            print(f"✅ GROKKED at step {grok_step}")
            return True
        else:
            # Check final loss
            for line in reversed(output_lines):
                if "- loss_valid_inter:" in line:
                    try:
                        final_loss = float(line.split(":")[1].strip())
                        print(f"❌ Failed (loss: {final_loss:.2e})")
                        return False
                    except:
                        continue
            print(f"❌ Failed (unknown)")
            return False

    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout")
        return False


def main():
    print("QUICK FROZEN SELECTOR TEST")
    print("=" * 50)

    results = {}

    # Test MUL with frozen multiplication selectors
    print("\n🔸 Testing MUL with freeze_O_selector_mul=True")
    modify_dag_defaults(freeze_div=False, freeze_mul=True)

    mul_results = []
    for seed in TEST_SEEDS:
        for interp_range, extrap_range in QUICK_RANGES:
            success = run_quick_test("mul", seed, interp_range, extrap_range)
            mul_results.append(success)

    # Test DIV with frozen division selectors
    print("\n🔸 Testing DIV with freeze_O_selectors_div=True")
    modify_dag_defaults(freeze_div=True, freeze_mul=False)

    div_results = []
    for seed in TEST_SEEDS:
        for interp_range, extrap_range in QUICK_RANGES:
            success = run_quick_test("div", seed, interp_range, extrap_range)
            div_results.append(success)

    # Summary
    print(f"\n" + "=" * 50)
    print("QUICK TEST SUMMARY")
    print("=" * 50)

    total_tests = len(mul_results) + len(div_results)
    total_success = sum(mul_results) + sum(div_results)

    mul_success_rate = sum(mul_results) / len(mul_results) if mul_results else 0
    div_success_rate = sum(div_results) / len(div_results) if div_results else 0
    overall_success_rate = total_success / total_tests if total_tests > 0 else 0

    print(
        f"MUL (frozen mul selectors): {mul_success_rate:.1%} ({sum(mul_results)}/{len(mul_results)})"
    )
    print(
        f"DIV (frozen div selectors): {div_success_rate:.1%} ({sum(div_results)}/{len(div_results)})"
    )
    print(f"Overall: {overall_success_rate:.1%} ({total_success}/{total_tests})")


if __name__ == "__main__":
    main()
