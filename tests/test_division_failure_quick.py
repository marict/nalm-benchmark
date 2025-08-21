#!/usr/bin/env python3
"""
Quick test demonstrating division training failure with very minimal setup.
"""

import re
import subprocess
import sys


def run_quick_test(operation: str, seed: int) -> dict:
    """Run a very quick training test."""
    cmd = [
        sys.executable,
        "experiments/single_layer_benchmark.py",
        "--layer-type",
        "DAG",
        "--operation",
        operation,
        "--input-size",
        "2",
        "--batch-size",
        "4",  # Very small
        "--max-iterations",
        "100",  # Very short
        "--learning-rate",
        "1e-3",
        "--log-interval",
        "50",
        "--interpolation-range",
        "[-2.0,2.0]",
        "--extrapolation-range",
        "[[-4.0,-2.0],[2.0,4.0]]",
        "--seed",
        str(seed),
        "--no-cuda",
        "--no-open-browser",
        "--no-save",
        "--note",
        f"quick_test_{operation}_seed{seed}",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
            return {"success": False, "error": result.stderr}

        # Parse for basic metrics
        output = result.stdout
        metrics = {}

        # Look for selector patterns in output
        selector_lines = [line for line in output.split("\n") if "values: [" in line]
        gate_lines = [line for line in output.split("\n") if "G: [" in line]

        # Parse gate values
        if gate_lines:
            try:
                g_line = gate_lines[0]
                g_str = g_line.split("G: ")[1]
                g_values = eval(g_str)
                metrics["gate_values"] = g_values
            except:
                pass

        # Parse selector values
        metrics["selector_patterns"] = []
        for line in selector_lines[:3]:  # First 3 patterns
            try:
                values_str = line.split("values: ")[1]
                values = eval(values_str)
                metrics["selector_patterns"].append(values)
            except:
                pass

        # Look for final training error
        train_lines = [
            line for line in output.split("\n") if "train " in line and "inter:" in line
        ]
        if train_lines:
            try:
                last_line = train_lines[-1]
                train_loss = float(last_line.split(": ")[1].split(",")[0])
                inter_error = float(last_line.split("inter: ")[1].split(",")[0])
                metrics["final_train_loss"] = train_loss
                metrics["final_inter_error"] = inter_error
            except:
                pass

        metrics["success"] = True
        return metrics

    except subprocess.TimeoutExpired:
        return {"success": False, "error": "timeout"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def analyze_patterns(metrics: dict, operation: str) -> dict:
    """Analyze if the patterns make sense for the operation."""
    analysis = {"issues": []}

    gate_values = metrics.get("gate_values", [])
    selector_patterns = metrics.get("selector_patterns", [])

    if operation == "div":
        # Division should use log domain (G ≈ 0) and select operands meaningfully
        if gate_values and len(gate_values) > 0:
            first_gate = gate_values[0]
            if first_gate > 0.3:  # Should be in log domain
                analysis["issues"].append(
                    f"Gate in wrong domain: {first_gate:.3f} (should be ≈0 for log)"
                )

        if selector_patterns and len(selector_patterns) > 0:
            first_pattern = selector_patterns[0]
            if len(first_pattern) >= 2:
                s1, s2 = first_pattern[0], first_pattern[1]
                if abs(s1) < 0.05 and abs(s2) < 0.05:
                    analysis["issues"].append(
                        f"Selectors too small: [{s1:.3f}, {s2:.3f}] (not learning)"
                    )

    elif operation == "add":
        # Addition should use linear domain (G ≈ 1)
        if gate_values and len(gate_values) > 0:
            first_gate = gate_values[0]
            if first_gate < 0.7:
                analysis["issues"].append(
                    f"Gate in wrong domain: {first_gate:.3f} (should be ≈1 for linear)"
                )

    return analysis


def main():
    """Quick division failure test."""
    print("Quick Division Failure Test")
    print("=" * 30)

    # Test addition first
    print("\n🧮 Testing Addition (should work):")
    add_result = run_quick_test("add", seed=0)

    if add_result["success"]:
        add_analysis = analyze_patterns(add_result, "add")
        print(f"  Gate values: {add_result.get('gate_values', 'N/A')}")
        print(
            f"  First selector: {add_result.get('selector_patterns', [['N/A']])[0][:2] if add_result.get('selector_patterns') else 'N/A'}"
        )
        print(f"  Final loss: {add_result.get('final_inter_error', 'N/A')}")
        print(f"  Issues: {len(add_analysis['issues'])} found")
        for issue in add_analysis["issues"][:2]:  # Show first 2 issues
            print(f"    - {issue}")
    else:
        print(f"  ❌ FAILED: {add_result['error']}")

    # Test division
    print(f"\n➗ Testing Division (likely to fail):")
    div_result = run_quick_test("div", seed=0)

    if div_result["success"]:
        div_analysis = analyze_patterns(div_result, "div")
        print(f"  Gate values: {div_result.get('gate_values', 'N/A')}")
        print(
            f"  First selector: {div_result.get('selector_patterns', [['N/A']])[0][:2] if div_result.get('selector_patterns') else 'N/A'}"
        )
        print(f"  Final loss: {div_result.get('final_inter_error', 'N/A')}")
        print(f"  Issues: {len(div_analysis['issues'])} found")
        for issue in div_analysis["issues"]:
            print(f"    - {issue}")

        # Summary comparison
        print(f"\n📊 Summary:")
        add_issues = (
            analyze_patterns(add_result, "add")["issues"]
            if add_result["success"]
            else ["Training failed"]
        )
        div_issues = div_analysis["issues"]

        print(f"  Addition issues: {len(add_issues)}")
        print(f"  Division issues: {len(div_issues)}")

        if len(div_issues) > len(add_issues):
            print(f"  ✅ CONFIRMED: Division has more training issues than addition")
        else:
            print(f"  🤔 UNEXPECTED: Division not clearly worse than addition")

    else:
        print(f"  ❌ FAILED: {div_result['error']}")


if __name__ == "__main__":
    main()
