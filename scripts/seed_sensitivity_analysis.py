#!/usr/bin/env python3
"""
Seed sensitivity analysis for arithmetic operations.

Tests grokking success rates across different seeds and operations to understand
initialization behavior and seed dependence patterns.
"""

import argparse
import json
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# Test seeds - using a diverse set to explore initialization space
TEST_SEEDS = [1, 7, 42, 99, 123, 221, 234, 345, 456, 789]

# Standard hyperparameters that have shown good results
STANDARD_HYPERPARAMS = {
    "layer_type": "DAG",
    "input_size": 2,
    "batch_size": 512,
    "max_iterations": 2000,  # Early cutoff for non-grokking detection
    "learning_rate": 1e-2,
    "interpolation_range": "[-2.0,2.0]",
    "extrapolation_range": "[[-6.0,-2.0],[2.0,6.0]]",
    "no_cuda": True,
    "log_interval": 100,
    "clip_grad_norm": 0.01,
    "no_open_browser": True,
}

# Grokking thresholds (early stopping triggers at 1e-10 for 100 iterations)
GROKKING_THRESHOLD = 1e-8  # Slightly relaxed from 1e-10 for analysis
SUCCESS_EARLY_STOP_PATTERN = "Early stopping at step"


def run_single_seed_test(operation: str, seed: int, max_iterations: int = 2000):
    """Run a single seed test for a specific operation."""

    cmd = ["python", "experiments/single_layer_benchmark.py"]

    # Add all standard hyperparameters
    for param, value in STANDARD_HYPERPARAMS.items():
        if param == "no_cuda" or param == "no_open_browser":
            if value:
                cmd.append(f'--{param.replace("_", "-")}')
        else:
            cmd.extend([f'--{param.replace("_", "-")}', str(value)])

    # Add operation and seed
    cmd.extend(["--operation", operation])
    cmd.extend(["--seed", str(seed)])
    cmd.extend(["--max-iterations", str(max_iterations)])

    print(f"🔬 Testing {operation.upper()} with seed {seed}")

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
            cwd="/Users/paul_curry/ai2/nalm-benchmark",
        )

        duration = time.time() - start_time

        # Parse output for grokking indicators
        output_lines = result.stdout.strip().split("\n")
        grokked = False
        early_stopped = False
        grok_step = None
        final_inter_loss = float("inf")
        final_extra_loss = float("inf")

        # Look for early stopping (indicates grokking)
        for line in output_lines:
            if SUCCESS_EARLY_STOP_PATTERN in line and "inter=" in line:
                early_stopped = True
                grokked = True
                try:
                    # Extract step number
                    step_part = line.split("Early stopping at step ")[1].split(":")[0]
                    grok_step = int(step_part)

                    # Extract final losses
                    inter_part = line.split("inter=")[1].split(",")[0]
                    extra_part = line.split("extra=")[1]
                    final_inter_loss = float(inter_part)
                    final_extra_loss = float(extra_part)

                    # Verify it's actually grokking (not just numerical error)
                    if final_inter_loss < GROKKING_THRESHOLD:
                        grokked = True
                    else:
                        grokked = False

                except (ValueError, IndexError):
                    grokked = False
                break

        # If no early stopping, check final performance
        if not early_stopped:
            # Look for the last training output
            for line in reversed(output_lines):
                if line.startswith("train ") and ":" in line:
                    try:
                        parts = line.split(":")
                        if len(parts) >= 3:
                            inter_loss = float(parts[2].split(",")[0].strip())
                            final_inter_loss = inter_loss
                            if final_inter_loss < GROKKING_THRESHOLD:
                                grokked = True
                            break
                    except (ValueError, IndexError):
                        continue

        success = result.returncode == 0

        return {
            "operation": operation,
            "seed": seed,
            "success": success,
            "grokked": grokked,
            "early_stopped": early_stopped,
            "grok_step": grok_step,
            "duration": duration,
            "final_inter_loss": final_inter_loss,
            "final_extra_loss": final_extra_loss,
            "stdout_excerpt": result.stdout[-2000:] if result.stdout else "",
            "stderr_excerpt": result.stderr[-1000:] if result.stderr else "",
        }

    except subprocess.TimeoutExpired:
        return {
            "operation": operation,
            "seed": seed,
            "success": False,
            "grokked": False,
            "early_stopped": False,
            "grok_step": None,
            "duration": time.time() - start_time,
            "final_inter_loss": float("inf"),
            "final_extra_loss": float("inf"),
            "stdout_excerpt": "TIMEOUT",
            "stderr_excerpt": "TIMEOUT",
        }
    except Exception as e:
        return {
            "operation": operation,
            "seed": seed,
            "success": False,
            "grokked": False,
            "early_stopped": False,
            "grok_step": None,
            "duration": time.time() - start_time,
            "final_inter_loss": float("inf"),
            "final_extra_loss": float("inf"),
            "stdout_excerpt": "",
            "stderr_excerpt": str(e),
        }


def analyze_results(results):
    """Analyze the results and compute success rates."""

    # Group by operation
    by_operation = {}
    for result in results:
        op = result["operation"]
        if op not in by_operation:
            by_operation[op] = []
        by_operation[op].append(result)

    analysis = {
        "total_experiments": len(results),
        "successful_runs": sum(1 for r in results if r["success"]),
        "operations": {},
    }

    for operation, op_results in by_operation.items():
        total_seeds = len(op_results)
        successful_runs = [r for r in op_results if r["success"]]
        grokked_runs = [r for r in op_results if r["grokked"]]

        # Calculate statistics
        success_rate = len(successful_runs) / total_seeds if total_seeds > 0 else 0
        grok_rate = len(grokked_runs) / total_seeds if total_seeds > 0 else 0
        grok_rate_given_success = (
            len(grokked_runs) / len(successful_runs) if successful_runs else 0
        )

        # Calculate average grok step for successful grokking
        grok_steps = [
            r["grok_step"] for r in grokked_runs if r["grok_step"] is not None
        ]
        avg_grok_step = np.mean(grok_steps) if grok_steps else None
        median_grok_step = np.median(grok_steps) if grok_steps else None

        # Find best and worst performing seeds
        successful_runs.sort(key=lambda x: x["final_inter_loss"])
        best_seed = successful_runs[0] if successful_runs else None
        worst_successful_seed = successful_runs[-1] if successful_runs else None

        # Seeds that grokked
        grokked_seeds = [r["seed"] for r in grokked_runs]
        failed_seeds = [r["seed"] for r in op_results if not r["success"]]
        non_grok_seeds = [
            r["seed"] for r in op_results if r["success"] and not r["grokked"]
        ]

        analysis["operations"][operation] = {
            "total_seeds": total_seeds,
            "successful_runs": len(successful_runs),
            "grokked_runs": len(grokked_runs),
            "success_rate": success_rate,
            "grok_rate": grok_rate,
            "grok_rate_given_success": grok_rate_given_success,
            "avg_grok_step": avg_grok_step,
            "median_grok_step": median_grok_step,
            "best_seed_result": best_seed,
            "worst_successful_result": worst_successful_seed,
            "grokked_seeds": grokked_seeds,
            "failed_seeds": failed_seeds,
            "non_grok_seeds": non_grok_seeds,
            "all_results": op_results,
        }

    return analysis


def print_summary(analysis):
    """Print a formatted summary of the analysis."""

    print("\n" + "=" * 80)
    print("🧬 SEED SENSITIVITY ANALYSIS RESULTS")
    print("=" * 80)
    print(f"Total experiments: {analysis['total_experiments']}")
    print(f"Successful runs: {analysis['successful_runs']}")
    print()

    # Sort operations by grok rate for display
    operations = list(analysis["operations"].keys())
    operations.sort(
        key=lambda op: analysis["operations"][op]["grok_rate"], reverse=True
    )

    for operation in operations:
        op_data = analysis["operations"][operation]

        print(f"📊 {operation.upper()} OPERATION:")
        print(f"   Total seeds tested: {op_data['total_seeds']}")
        print(
            f"   Success rate: {op_data['success_rate']*100:.1f}% ({op_data['successful_runs']}/{op_data['total_seeds']})"
        )
        print(
            f"   Grok rate (all seeds): {op_data['grok_rate']*100:.1f}% ({op_data['grokked_runs']}/{op_data['total_seeds']})"
        )
        print(
            f"   Grok rate (successful only): {op_data['grok_rate_given_success']*100:.1f}% ({op_data['grokked_runs']}/{op_data['successful_runs']})"
        )

        if op_data["avg_grok_step"] is not None:
            print(f"   Average grok step: {op_data['avg_grok_step']:.0f}")
            print(f"   Median grok step: {op_data['median_grok_step']:.0f}")

        print(f"   Grokked seeds: {op_data['grokked_seeds']}")
        print(f"   Non-grok seeds: {op_data['non_grok_seeds']}")

        if op_data["failed_seeds"]:
            print(f"   Failed seeds: {op_data['failed_seeds']}")

        if op_data["best_seed_result"]:
            best = op_data["best_seed_result"]
            print(
                f"   Best seed: {best['seed']} (loss: {best['final_inter_loss']:.2e})"
            )

        print()

    # Overall insights
    print("🔍 KEY INSIGHTS:")

    # Find most/least seed-dependent operations
    grok_rates = [
        (op, data["grok_rate"]) for op, data in analysis["operations"].items()
    ]
    grok_rates.sort(key=lambda x: x[1], reverse=True)

    if grok_rates:
        most_reliable = grok_rates[0]
        least_reliable = grok_rates[-1]

        print(
            f"   Most reliable: {most_reliable[0].upper()} ({most_reliable[1]*100:.1f}% grok rate)"
        )
        print(
            f"   Least reliable: {least_reliable[0].upper()} ({least_reliable[1]*100:.1f}% grok rate)"
        )

    # Find "golden seeds" that work across operations
    all_grokked_seeds = {}
    for op, data in analysis["operations"].items():
        for seed in data["grokked_seeds"]:
            if seed not in all_grokked_seeds:
                all_grokked_seeds[seed] = []
            all_grokked_seeds[seed].append(op)

    golden_seeds = [
        (seed, ops) for seed, ops in all_grokked_seeds.items() if len(ops) >= 3
    ]
    if golden_seeds:
        golden_seeds.sort(key=lambda x: len(x[1]), reverse=True)
        print(f"   Golden seeds (work on 3+ ops): {[seed for seed, _ in golden_seeds]}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Seed sensitivity analysis for arithmetic operations"
    )
    parser.add_argument(
        "--operations",
        nargs="+",
        default=["add", "sub", "mul", "div"],
        help="Operations to test (default: all)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=TEST_SEEDS,
        help=f"Seeds to test (default: {TEST_SEEDS})",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=2000,
        help="Max iterations per test (default: 2000)",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=4,
        help="Maximum parallel processes (default: 4)",
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default="seed_sensitivity_results.json",
        help="Results file name",
    )

    args = parser.parse_args()

    # Create results directory
    results_dir = Path("seed_analysis_results")
    results_dir.mkdir(exist_ok=True)

    # Generate all test combinations
    test_combinations = [
        (operation, seed) for operation in args.operations for seed in args.seeds
    ]

    print(f"🧪 Starting seed sensitivity analysis...")
    print(f"   Operations: {args.operations}")
    print(f"   Seeds: {args.seeds}")
    print(f"   Total combinations: {len(test_combinations)}")
    print(f"   Max iterations per test: {args.max_iterations}")
    print(f"   Parallel processes: {args.max_parallel}")
    print()

    results = []
    completed = 0

    # Run experiments in parallel
    with ProcessPoolExecutor(max_workers=args.max_parallel) as executor:
        # Submit all experiments
        future_to_test = {
            executor.submit(
                run_single_seed_test, operation, seed, args.max_iterations
            ): (operation, seed)
            for operation, seed in test_combinations
        }

        # Collect results as they complete
        for future in as_completed(future_to_test):
            operation, seed = future_to_test[future]

            try:
                result = future.result()
                results.append(result)
                completed += 1

                # Print progress
                status = (
                    "✅ GROK"
                    if result["grokked"]
                    else ("⚪ OK" if result["success"] else "❌ FAIL")
                )
                step_info = f" @{result['grok_step']}" if result["grok_step"] else ""
                loss_info = (
                    f" (loss: {result['final_inter_loss']:.2e})"
                    if result["final_inter_loss"] != float("inf")
                    else ""
                )

                print(f"{status} {operation.upper()}/seed{seed}{step_info}{loss_info}")
                print(
                    f"   Progress: {completed}/{len(test_combinations)} ({completed/len(test_combinations)*100:.1f}%)"
                )
                print()

                # Save intermediate results
                results_file = results_dir / args.results_file
                with open(results_file, "w") as f:
                    json.dump(
                        {
                            "metadata": {
                                "operations": args.operations,
                                "seeds": args.seeds,
                                "max_iterations": args.max_iterations,
                                "hyperparameters": STANDARD_HYPERPARAMS,
                                "grokking_threshold": GROKKING_THRESHOLD,
                            },
                            "results": results,
                        },
                        f,
                        indent=2,
                        default=str,
                    )

            except Exception as e:
                print(f"❌ Exception in {operation}/seed{seed}: {e}")

    # Analyze results
    analysis = analyze_results(results)

    # Save final results with analysis
    results_file = results_dir / args.results_file
    with open(results_file, "w") as f:
        json.dump(
            {
                "metadata": {
                    "operations": args.operations,
                    "seeds": args.seeds,
                    "max_iterations": args.max_iterations,
                    "hyperparameters": STANDARD_HYPERPARAMS,
                    "grokking_threshold": GROKKING_THRESHOLD,
                },
                "results": results,
                "analysis": analysis,
            },
            f,
            indent=2,
            default=str,
        )

    # Print summary
    print_summary(analysis)

    print(f"📁 Results saved to: {results_file}")

    # Return success if at least some seeds grokked for each operation
    min_grok_rate = min(data["grok_rate"] for data in analysis["operations"].values())
    return min_grok_rate > 0.1  # At least 10% success rate across all operations


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
