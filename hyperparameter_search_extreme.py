#!/usr/bin/env python3

import argparse
import json
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# EXTREME hyperparameter search - going to the absolute limits
# Based on the gradient explosion analysis, we need RADICAL intervention

EXTREME_SEARCH_SPACE = {
    # Microscopic learning rates
    "learning_rate": [1e-7, 5e-7, 1e-6, 2e-6],
    # Extremely aggressive gradient clipping
    "clip_grad_norm": [0.0001, 0.0005, 0.001],
    # Smaller batches for more frequent updates
    "batch_size": [16, 32, 64],
    # Very restrictive target filtering
    "max_target_magnitude": [1.0, 2.0, 5.0],
    # Many different seeds to find lucky initialization
    "seed": [
        1,
        7,
        42,
        99,
        123,
        234,
        345,
        456,
        567,
        678,
        789,
        890,
        1001,
        1123,
        1234,
        2024,
        3141,
        5678,
        7890,
        9999,
    ],
    # Minimal DAG complexity
    "num_subsets": [1],
    # Maximum division regularization
    "div_regularizer": [0.1, 0.5, 1.0],
}

# Nuclear option configurations - going to absolute extremes
NUCLEAR_CONFIGS = [
    # Ultra-micro learning rate with maximum clipping
    {
        "learning_rate": 1e-7,
        "clip_grad_norm": 0.0001,
        "batch_size": 16,
        "max_target_magnitude": 1.0,
        "num_subsets": 1,
        "div_regularizer": 1.0,
    },
    # Slightly less extreme but still very aggressive
    {
        "learning_rate": 5e-7,
        "clip_grad_norm": 0.0005,
        "batch_size": 32,
        "max_target_magnitude": 2.0,
        "num_subsets": 1,
        "div_regularizer": 0.5,
    },
    # The "reasonable extreme"
    {
        "learning_rate": 1e-6,
        "clip_grad_norm": 0.001,
        "batch_size": 64,
        "max_target_magnitude": 5.0,
        "num_subsets": 1,
        "div_regularizer": 0.1,
    },
]


# Generate many variations with different seeds
def generate_seeded_configs():
    """Generate the same extreme configs with many different seeds"""
    base_configs = NUCLEAR_CONFIGS
    seeded_configs = []

    seeds_to_try = [
        1,
        7,
        42,
        99,
        123,
        234,
        345,
        456,
        567,
        678,
        789,
        890,
        1001,
        1123,
        1234,
        2024,
        3141,
        5678,
        7890,
        9999,
    ]

    for seed in seeds_to_try:
        for config in base_configs:
            new_config = config.copy()
            new_config["seed"] = seed
            seeded_configs.append(new_config)

    return seeded_configs


def generate_random_extreme_configs(n_configs=50):
    """Generate random extreme configurations"""
    configs = []

    for _ in range(n_configs):
        config = {}
        for param, values in EXTREME_SEARCH_SPACE.items():
            config[param] = random.choice(values)
        configs.append(config)

    return configs


def run_single_experiment(config, experiment_id, max_iterations=10000):
    """Run a single EXTREME experiment"""

    # Build command with EXTREME patience
    cmd = [
        "python",
        "experiments/single_layer_benchmark.py",
        "--layer-type",
        "DAG",
        "--operation",
        "div",
        "--input-size",
        "2",
        "--interpolation-range",
        "[-2.0,2.0]",
        "--extrapolation-range",
        "[[-6.0,-2.0],[2.0,6.0]]",
        "--no-cuda",
        "--max-iterations",
        str(max_iterations),
        "--log-interval",
        "1000",  # Less frequent logging for long runs
        "--no-open-browser",
        "--note",
        f"extreme_search_{experiment_id}",
    ]

    # Add hyperparameters to command
    for param, value in config.items():
        if value is not None:
            cmd.extend([f'--{param.replace("_", "-")}', str(value)])

    print(f"🚀 EXTREME Experiment {experiment_id}: {config}")

    start_time = time.time()

    try:
        # Run with MUCH longer timeout for extreme runs
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1200,  # 20 minute timeout for 10K iterations
            cwd="/Users/paul_curry/ai2/nalm-benchmark",
        )

        duration = time.time() - start_time

        # Parse the output to extract final losses
        output_lines = result.stdout.strip().split("\n")
        final_losses = {}
        grokked = False
        early_stopped = False
        best_inter_loss = float("inf")

        # Look through all training lines to find the best achieved loss
        for line in output_lines:
            if "Early stopping at step" in line:
                early_stopped = True
                try:
                    if "inter=" in line and "extra=" in line:
                        inter_part = line.split("inter=")[1].split(",")[0]
                        extra_part = line.split("extra=")[1]
                        inter_loss = float(inter_part)
                        extra_loss = float(extra_part)
                        final_losses = {
                            "train": inter_loss,
                            "interpolation": inter_loss,
                            "extrapolation": extra_loss,
                        }
                        best_inter_loss = min(best_inter_loss, inter_loss)
                        if inter_loss < 1e-3:  # More relaxed grokking threshold
                            grokked = True
                        break
                except (ValueError, IndexError):
                    continue

            elif line.startswith("train ") and ":" in line:
                try:
                    parts = line.split(":")
                    if len(parts) >= 4:
                        train_loss = float(parts[1].split(",")[0].strip())
                        inter_loss = float(parts[2].split(",")[0].strip())
                        extra_loss = float(parts[3].strip())

                        best_inter_loss = min(best_inter_loss, inter_loss)

                        final_losses = {
                            "train": train_loss,
                            "interpolation": inter_loss,
                            "extrapolation": extra_loss,
                        }

                        if inter_loss < 1e-3:  # More relaxed grokking threshold
                            grokked = True
                except (ValueError, IndexError):
                    continue

        # Check if we have valid losses
        if not final_losses:
            final_losses = {
                "train": float("inf"),
                "interpolation": (
                    best_inter_loss if best_inter_loss != float("inf") else float("inf")
                ),
                "extrapolation": float("inf"),
            }
        else:
            # Update with best loss seen
            final_losses["best_interpolation"] = best_inter_loss

        success = result.returncode == 0

        return {
            "experiment_id": experiment_id,
            "config": config,
            "success": success,
            "grokked": grokked,
            "early_stopped": early_stopped,
            "duration": duration,
            "final_losses": final_losses,
            "best_interpolation_loss": best_inter_loss,
            "stdout": result.stdout[-3000:] if result.stdout else "",  # Last 3000 chars
            "stderr": result.stderr[-1000:] if result.stderr else "",  # Last 1000 chars
        }

    except subprocess.TimeoutExpired:
        return {
            "experiment_id": experiment_id,
            "config": config,
            "success": False,
            "grokked": False,
            "early_stopped": False,
            "duration": time.time() - start_time,
            "final_losses": {
                "train": float("inf"),
                "interpolation": float("inf"),
                "extrapolation": float("inf"),
            },
            "best_interpolation_loss": float("inf"),
            "stdout": "",
            "stderr": "Timeout expired",
        }
    except Exception as e:
        return {
            "experiment_id": experiment_id,
            "config": config,
            "success": False,
            "grokked": False,
            "early_stopped": False,
            "duration": time.time() - start_time,
            "final_losses": {
                "train": float("inf"),
                "interpolation": float("inf"),
                "extrapolation": float("inf"),
            },
            "best_interpolation_loss": float("inf"),
            "stdout": "",
            "stderr": str(e),
        }


def main():
    parser = argparse.ArgumentParser(
        description="EXTREME hyperparameter search - nuclear option"
    )
    parser.add_argument(
        "--max-experiments",
        type=int,
        default=30,
        help="Maximum number of experiments to run",
    )
    parser.add_argument(
        "--max-parallel", type=int, default=4, help="Maximum parallel processes"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10000,
        help="Maximum iterations per experiment",
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default="div_extreme_search_results.json",
        help="Results file",
    )
    parser.add_argument(
        "--nuclear-only",
        action="store_true",
        help="Only run nuclear option configurations",
    )
    parser.add_argument(
        "--seeded-configs",
        action="store_true",
        help="Run seeded versions of nuclear configs",
    )

    args = parser.parse_args()

    # Create results directory
    results_dir = Path("hyperparameter_search_results")
    results_dir.mkdir(exist_ok=True)

    # Generate configurations
    if args.nuclear_only:
        configs = NUCLEAR_CONFIGS[: args.max_experiments]
        print(f"🚀 Running {len(configs)} NUCLEAR configurations")
    elif args.seeded_configs:
        configs = generate_seeded_configs()[: args.max_experiments]
        print(
            f"🚀 Running {len(configs)} SEEDED NUCLEAR configurations (multiple seeds)"
        )
    else:
        base_configs = NUCLEAR_CONFIGS + generate_seeded_configs()[:15]
        configs = base_configs + generate_random_extreme_configs(
            args.max_experiments - len(base_configs)
        )
        configs = configs[: args.max_experiments]
        print(f"🚀 Running {len(configs)} EXTREME configurations")

    print("⚠️  WARNING: Using EXTREME hyperparameters:")
    print("   - Learning rates as low as 1e-7")
    print("   - Gradient clipping as tight as 0.0001")
    print("   - Target magnitude filtering as strict as 1.0")
    print("   - Up to 10,000 iterations per run")
    print("   - Multiple seeds for each config")
    print()

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    results = []
    grokked_configs = []
    excellent_progress_configs = []  # < 1.0
    good_progress_configs = []  # < 3.0

    print(
        f"Starting EXTREME hyperparameter search with {args.max_parallel} parallel processes..."
    )

    # Run experiments in parallel
    with ProcessPoolExecutor(max_workers=args.max_parallel) as executor:
        # Submit all experiments
        future_to_config = {
            executor.submit(run_single_experiment, config, i, args.max_iterations): (
                config,
                i,
            )
            for i, config in enumerate(configs)
        }

        # Collect results as they complete
        for future in as_completed(future_to_config):
            config, experiment_id = future_to_config[future]

            try:
                result = future.result()
                results.append(result)

                best_loss = result["best_interpolation_loss"]

                # Check if this configuration achieved grokking
                if result["grokked"]:
                    grokked_configs.append(result)
                    print(f"🎉 GROKKING ACHIEVED! Experiment {experiment_id}")
                    print(f"   Best interpolation loss: {best_loss:.2e}")
                    print(f"   Config: {result['config']}")
                    print(f"   Duration: {result['duration']:.1f}s")
                elif result["success"] and best_loss < 1.0:
                    excellent_progress_configs.append(result)
                    print(f"🌟 Excellent progress - Experiment {experiment_id}")
                    print(f"   Best interpolation loss: {best_loss:.4f}")
                    print(f"   Config: {result['config']}")
                elif result["success"] and best_loss < 3.0:
                    good_progress_configs.append(result)
                    print(f"📈 Good progress - Experiment {experiment_id}")
                    print(f"   Best interpolation loss: {best_loss:.4f}")
                    print(f"   Config: {result['config']}")
                elif result["success"]:
                    print(f"⚪ Completed - Experiment {experiment_id}")
                    print(f"   Best interpolation loss: {best_loss:.4f}")
                    print(f"   Config: {result['config']}")
                else:
                    print(f"❌ Failed - Experiment {experiment_id}")

                # Print progress
                completed = len(results)
                print(
                    f"Progress: {completed}/{len(configs)} ({completed/len(configs)*100:.1f}%)"
                )
                print(
                    f"Grokking: {len(grokked_configs)}, Excellent: {len(excellent_progress_configs)}, Good: {len(good_progress_configs)}"
                )
                print()

                # Save intermediate results
                results_file = results_dir / args.results_file
                with open(results_file, "w") as f:
                    json.dump(
                        {
                            "problem_setup": {
                                "interpolation_range": "[-2.0,2.0]",
                                "extrapolation_range": "[[-6.0,-2.0],[2.0,6.0]]",
                                "note": "EXTREME hyperparameter search - nuclear option",
                            },
                            "search_space": EXTREME_SEARCH_SPACE,
                            "nuclear_configs": NUCLEAR_CONFIGS,
                            "results": results,
                            "grokked_configs": grokked_configs,
                            "excellent_progress_configs": excellent_progress_configs,
                            "good_progress_configs": good_progress_configs,
                            "summary": {
                                "total_experiments": len(results),
                                "successful_experiments": sum(
                                    1 for r in results if r["success"]
                                ),
                                "grokking_experiments": len(grokked_configs),
                                "excellent_progress_experiments": len(
                                    excellent_progress_configs
                                ),
                                "good_progress_experiments": len(good_progress_configs),
                                "grokking_rate": (
                                    len(grokked_configs) / len(results)
                                    if results
                                    else 0
                                ),
                            },
                        },
                        f,
                        indent=2,
                        default=str,
                    )

            except Exception as e:
                print(f"❌ Experiment {experiment_id} failed with exception: {e}")
                results.append(
                    {
                        "experiment_id": experiment_id,
                        "config": config,
                        "success": False,
                        "grokked": False,
                        "early_stopped": False,
                        "duration": 0,
                        "final_losses": {
                            "train": float("inf"),
                            "interpolation": float("inf"),
                            "extrapolation": float("inf"),
                        },
                        "best_interpolation_loss": float("inf"),
                        "stdout": "",
                        "stderr": str(e),
                    }
                )

    # Final summary
    print("\n" + "=" * 60)
    print("🚀 EXTREME HYPERPARAMETER SEARCH RESULTS SUMMARY")
    print("=" * 60)
    print(f"Problem setup: HARD interpolation [-2,2], extrapolation [[-6,-2],[2,6]]")
    print(f"Total experiments: {len(results)}")
    print(f"Successful experiments: {sum(1 for r in results if r['success'])}")
    print(f"🎉 Grokking experiments (< 1e-3): {len(grokked_configs)}")
    print(f"🌟 Excellent progress (< 1.0): {len(excellent_progress_configs)}")
    print(f"📈 Good progress (< 3.0): {len(good_progress_configs)}")

    # Show best results
    all_successful = [r for r in results if r["success"]]
    if all_successful:
        all_successful.sort(key=lambda x: x["best_interpolation_loss"])

        print(f"\n🏆 TOP PERFORMING CONFIGURATIONS:")
        for i, result in enumerate(all_successful[:10]):  # Top 10
            status = "🎉 GROKKED" if result["grokked"] else "📊 PROGRESS"
            print(f"\n{status} Config {i+1}:")
            print(f"  Best Interpolation Loss: {result['best_interpolation_loss']:.4f}")
            print(
                f"  Final Interpolation Loss: {result['final_losses']['interpolation']:.4f}"
            )
            print(f"  Duration: {result['duration']:.1f}s")
            print(f"  Seed: {result['config']['seed']}")
            print(f"  Config: {result['config']}")

    print(f"\nResults saved to: {results_dir / args.results_file}")

    return len(grokked_configs) > 0 or len(excellent_progress_configs) > 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
