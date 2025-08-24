#!/usr/bin/env python3
"""
Division grokking optimization - focused hyperparameter search.

Goal: Find hyperparameter configurations that maximize division grokking success rates
within 2000 iterations, building on the seed sensitivity analysis insights.
"""

import argparse
import itertools
import json
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# Seeds that showed some division potential from previous analysis
# Include seed 223 (user's working seed) and other promising candidates
DIVISION_TEST_SEEDS = [221, 345, 223, 1, 7, 99, 123, 456, 789]

# Base hyperparameters (your working configuration)
BASE_HYPERPARAMS = {
    "layer_type": "DAG",
    "operation": "div",
    "input_size": 2,
    "interpolation_range": "[-2.0,2.0]",
    "extrapolation_range": "[[-6.0,-2.0],[2.0,6.0]]",
    "no_cuda": True,
    "max_iterations": 2000,
    "log_interval": 200,
    "no_open_browser": True,
}

# Hyperparameter search space focused on division optimization
SEARCH_SPACE = {
    # Learning rates - explore around your working 1e-2
    "learning_rate": [5e-3, 8e-3, 1e-2, 1.2e-2, 1.5e-2],
    # Batch sizes - smaller might help with division precision
    "batch_size": [256, 512, 768],
    # Gradient clipping - critical for division stability
    "clip_grad_norm": [0.005, 0.01, 0.02, 0.05],
    # Target magnitude filtering - might help division convergence
    "max_target_magnitude": [None, 50.0, 100.0],
}

# Promising configurations based on division-specific insights
PROMISING_CONFIGS = [
    # Your working config as baseline
    {
        "learning_rate": 1e-2,
        "batch_size": 512,
        "clip_grad_norm": 0.01,
        "max_target_magnitude": None,
    },
    # Slightly higher learning rate + tighter clipping
    {
        "learning_rate": 1.2e-2,
        "batch_size": 512,
        "clip_grad_norm": 0.005,
        "max_target_magnitude": None,
    },
    # Lower batch size for more frequent updates
    {
        "learning_rate": 1e-2,
        "batch_size": 256,
        "clip_grad_norm": 0.01,
        "max_target_magnitude": None,
    },
    # Target magnitude filtering
    {
        "learning_rate": 1e-2,
        "batch_size": 512,
        "clip_grad_norm": 0.01,
        "max_target_magnitude": 50.0,
    },
    # Conservative approach
    {
        "learning_rate": 8e-3,
        "batch_size": 768,
        "clip_grad_norm": 0.02,
        "max_target_magnitude": 100.0,
    },
]

GROKKING_THRESHOLD = 1e-8
SUCCESS_EARLY_STOP_PATTERN = "Early stopping at step"


def run_division_test(config, config_id, test_seeds):
    """Run division test across multiple seeds for a single configuration."""

    results = []

    for seed in test_seeds:
        cmd = ["python", "experiments/single_layer_benchmark.py"]

        # Add base hyperparameters
        for param, value in BASE_HYPERPARAMS.items():
            if param in ["no_cuda", "no_open_browser"]:
                if value:
                    cmd.append(f'--{param.replace("_", "-")}')
            else:
                cmd.extend([f'--{param.replace("_", "-")}', str(value)])

        # Add configuration hyperparameters
        for param, value in config.items():
            if value is not None:
                param_name = param.replace("_", "-")
                cmd.extend([f"--{param_name}", str(value)])

        # Add seed
        cmd.extend(["--seed", str(seed)])

        print(f"🧪 Config {config_id}, seed {seed}: {config}")

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
                    try:
                        # Extract step number
                        step_part = line.split("Early stopping at step ")[1].split(":")[
                            0
                        ]
                        grok_step = int(step_part)

                        # Extract final losses
                        inter_part = line.split("inter=")[1].split(",")[0]
                        extra_part = line.split("extra=")[1]
                        final_inter_loss = float(inter_part)
                        final_extra_loss = float(extra_part)

                        # Verify it's actually grokking
                        if final_inter_loss < GROKKING_THRESHOLD:
                            grokked = True
                        else:
                            grokked = False

                    except (ValueError, IndexError):
                        grokked = False
                    break

            # If no early stopping, check final performance
            if not early_stopped:
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

            result_data = {
                "config_id": config_id,
                "config": config,
                "seed": seed,
                "success": success,
                "grokked": grokked,
                "early_stopped": early_stopped,
                "grok_step": grok_step,
                "duration": duration,
                "final_inter_loss": final_inter_loss,
                "final_extra_loss": final_extra_loss,
            }

            # Print immediate result
            if grokked:
                print(f"  🎉 GROKKED @{grok_step} (loss: {final_inter_loss:.2e})")
            elif success:
                print(f"  ⚪ No grok (loss: {final_inter_loss:.2e})")
            else:
                print(f"  ❌ Failed")

            results.append(result_data)

        except subprocess.TimeoutExpired:
            results.append(
                {
                    "config_id": config_id,
                    "config": config,
                    "seed": seed,
                    "success": False,
                    "grokked": False,
                    "early_stopped": False,
                    "grok_step": None,
                    "duration": time.time() - start_time,
                    "final_inter_loss": float("inf"),
                    "final_extra_loss": float("inf"),
                }
            )
            print(f"  ⏰ Timeout")

        except Exception as e:
            results.append(
                {
                    "config_id": config_id,
                    "config": config,
                    "seed": seed,
                    "success": False,
                    "grokked": False,
                    "early_stopped": False,
                    "grok_step": None,
                    "duration": time.time() - start_time,
                    "final_inter_loss": float("inf"),
                    "final_extra_loss": float("inf"),
                }
            )
            print(f"  💥 Error: {e}")

    return results


def analyze_config_results(all_results):
    """Analyze results by configuration to find best performing setups."""

    # Group results by configuration
    by_config = {}
    for result in all_results:
        config_id = result["config_id"]
        if config_id not in by_config:
            by_config[config_id] = {"config": result["config"], "results": []}
        by_config[config_id]["results"].append(result)

    # Analyze each configuration
    config_analysis = []

    for config_id, data in by_config.items():
        config = data["config"]
        results = data["results"]

        total_seeds = len(results)
        successful_runs = [r for r in results if r["success"]]
        grokked_runs = [r for r in results if r["grokked"]]

        success_rate = len(successful_runs) / total_seeds if total_seeds > 0 else 0
        grok_rate = len(grokked_runs) / total_seeds if total_seeds > 0 else 0

        # Calculate average grok step
        grok_steps = [
            r["grok_step"] for r in grokked_runs if r["grok_step"] is not None
        ]
        avg_grok_step = np.mean(grok_steps) if grok_steps else None

        # Find successful seeds
        grokked_seeds = [r["seed"] for r in grokked_runs]

        config_analysis.append(
            {
                "config_id": config_id,
                "config": config,
                "total_seeds": total_seeds,
                "success_rate": success_rate,
                "grok_rate": grok_rate,
                "grokked_count": len(grokked_runs),
                "avg_grok_step": avg_grok_step,
                "grokked_seeds": grokked_seeds,
                "all_results": results,
            }
        )

    # Sort by grok rate (primary) and avg grok step (secondary)
    config_analysis.sort(
        key=lambda x: (
            x["grok_rate"],
            -x["avg_grok_step"] if x["avg_grok_step"] else 0,
        ),
        reverse=True,
    )

    return config_analysis


def print_analysis(config_analysis):
    """Print formatted analysis of the hyperparameter search."""

    print("\n" + "=" * 80)
    print("🎯 DIVISION GROKKING OPTIMIZATION RESULTS")
    print("=" * 80)

    for i, analysis in enumerate(config_analysis):
        config_id = analysis["config_id"]
        config = analysis["config"]
        grok_rate = analysis["grok_rate"] * 100
        grok_count = analysis["grokked_count"]
        total = analysis["total_seeds"]

        print(
            f"\n📊 CONFIG #{config_id} - Grok Rate: {grok_rate:.1f}% ({grok_count}/{total})"
        )
        print(f"   Config: {config}")

        if analysis["avg_grok_step"]:
            print(f"   Avg grok step: {analysis['avg_grok_step']:.0f}")

        if analysis["grokked_seeds"]:
            print(f"   Grokked seeds: {analysis['grokked_seeds']}")

    # Highlight best configuration
    if config_analysis:
        best = config_analysis[0]
        print(f"\n🏆 BEST CONFIGURATION:")
        print(
            f"   Grok rate: {best['grok_rate']*100:.1f}% ({best['grokked_count']}/{best['total_seeds']})"
        )
        print(f"   Config: {best['config']}")
        if best["avg_grok_step"]:
            print(f"   Avg grok step: {best['avg_grok_step']:.0f}")


def generate_random_configs(n_configs=15):
    """Generate additional random configurations from search space."""
    configs = []

    for _ in range(n_configs):
        config = {}
        for param, values in SEARCH_SPACE.items():
            config[param] = np.random.choice(values)
        configs.append(config)

    return configs


def main():
    parser = argparse.ArgumentParser(
        description="Division grokking hyperparameter optimization"
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=20,
        help="Maximum configurations to test (default: 20)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DIVISION_TEST_SEEDS,
        help=f"Seeds to test (default: {DIVISION_TEST_SEEDS})",
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default="division_optimization_results.json",
        help="Results file name",
    )
    parser.add_argument(
        "--promising-only",
        action="store_true",
        help="Only test promising configurations (no random search)",
    )

    args = parser.parse_args()

    # Create results directory
    results_dir = Path("division_optimization_results")
    results_dir.mkdir(exist_ok=True)

    # Generate configurations to test
    if args.promising_only:
        configs_to_test = PROMISING_CONFIGS[: args.max_configs]
    else:
        promising_count = min(len(PROMISING_CONFIGS), args.max_configs // 2)
        random_count = args.max_configs - promising_count
        configs_to_test = PROMISING_CONFIGS[:promising_count] + generate_random_configs(
            random_count
        )
        configs_to_test = configs_to_test[: args.max_configs]

    print(f"🚀 Starting division grokking optimization...")
    print(f"   Total configurations: {len(configs_to_test)}")
    print(f"   Seeds per config: {len(args.seeds)}")
    print(f"   Total experiments: {len(configs_to_test) * len(args.seeds)}")
    print(f"   Target: Maximize division grokking within 2000 iterations")
    print()

    all_results = []

    # Test each configuration
    for config_id, config in enumerate(configs_to_test):
        print(f"🧪 Testing Config #{config_id}: {config}")

        # Run tests for this configuration across all seeds
        config_results = run_division_test(config, config_id, args.seeds)
        all_results.extend(config_results)

        # Calculate and show immediate config summary
        grokked_count = sum(1 for r in config_results if r["grokked"])
        total_seeds = len(config_results)
        grok_rate = grokked_count / total_seeds * 100

        print(
            f"   Config #{config_id} summary: {grok_rate:.1f}% grok rate ({grokked_count}/{total_seeds})"
        )
        print()

        # Save intermediate results
        results_file = results_dir / args.results_file
        with open(results_file, "w") as f:
            json.dump(
                {
                    "metadata": {
                        "base_hyperparams": BASE_HYPERPARAMS,
                        "search_space": SEARCH_SPACE,
                        "test_seeds": args.seeds,
                        "max_iterations": 2000,
                        "grokking_threshold": GROKKING_THRESHOLD,
                    },
                    "all_results": all_results,
                },
                f,
                indent=2,
                default=str,
            )

    # Analyze results
    config_analysis = analyze_config_results(all_results)

    # Save final results with analysis
    results_file = results_dir / args.results_file
    with open(results_file, "w") as f:
        json.dump(
            {
                "metadata": {
                    "base_hyperparams": BASE_HYPERPARAMS,
                    "search_space": SEARCH_SPACE,
                    "test_seeds": args.seeds,
                    "max_iterations": 2000,
                    "grokking_threshold": GROKKING_THRESHOLD,
                },
                "all_results": all_results,
                "config_analysis": config_analysis,
            },
            f,
            indent=2,
            default=str,
        )

    # Print analysis
    print_analysis(config_analysis)

    print(f"\n📁 Results saved to: {results_file}")

    # Return success if best configuration achieved >50% grok rate
    if config_analysis:
        best_grok_rate = config_analysis[0]["grok_rate"]
        return best_grok_rate > 0.5

    return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
