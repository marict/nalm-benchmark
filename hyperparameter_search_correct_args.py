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

# Corrected hyperparameter search with the actual difficult setup:
# --interpolation-range "[-2.0,2.0]"  (includes negatives and near-zero)
# --extrapolation-range "[[-6.0,-2.0],[2.0,6.0]]"  (much harder extrapolation)

CORRECTED_SEARCH_SPACE = {
    # Even more aggressive parameters for the harder problem
    "learning_rate": [1e-6, 5e-6, 1e-5, 2e-5, 5e-5],
    "clip_grad_norm": [0.001, 0.005, 0.01, 0.05],
    "batch_size": [64, 128, 256],
    "max_target_magnitude": [5.0, 10.0, 20.0, 50.0],
    # Try different seeds since grokking can be seed-dependent
    "seed": [42, 123, 456, 789, 1123, 2024, 3141, 5678],
    # DAG-specific parameters
    "num_subsets": [1, 2],
    # Since regularizers don't work, focus on what does
    "div_regularizer": [0.001, 0.01, 0.1],
}

# Ultra-conservative configs for the harder problem
ULTRA_CONSERVATIVE_CONFIGS = [
    {
        "learning_rate": 1e-6,
        "clip_grad_norm": 0.001,
        "batch_size": 64,
        "max_target_magnitude": 5.0,
        "seed": 42,
        "num_subsets": 1,
        "div_regularizer": 0.1,
    },
    {
        "learning_rate": 5e-6,
        "clip_grad_norm": 0.005,
        "batch_size": 128,
        "max_target_magnitude": 10.0,
        "seed": 123,
        "num_subsets": 1,
        "div_regularizer": 0.01,
    },
    {
        "learning_rate": 1e-5,
        "clip_grad_norm": 0.01,
        "batch_size": 128,
        "max_target_magnitude": 10.0,
        "seed": 456,
        "num_subsets": 2,
        "div_regularizer": 0.001,
    },
    # Try some different seeds with same good config
    {
        "learning_rate": 1e-5,
        "clip_grad_norm": 0.01,
        "batch_size": 128,
        "max_target_magnitude": 10.0,
        "seed": 789,
        "num_subsets": 2,
        "div_regularizer": 0.001,
    },
    {
        "learning_rate": 1e-5,
        "clip_grad_norm": 0.01,
        "batch_size": 128,
        "max_target_magnitude": 10.0,
        "seed": 1123,
        "num_subsets": 2,
        "div_regularizer": 0.001,
    },
    {
        "learning_rate": 2e-5,
        "clip_grad_norm": 0.01,
        "batch_size": 256,
        "max_target_magnitude": 20.0,
        "seed": 2024,
        "num_subsets": 1,
        "div_regularizer": 0.01,
    },
]


def generate_random_configs(n_configs=30):
    """Generate random hyperparameter configurations"""
    configs = []

    for _ in range(n_configs):
        config = {}
        for param, values in CORRECTED_SEARCH_SPACE.items():
            config[param] = random.choice(values)
        configs.append(config)

    return configs


def run_single_experiment(config, experiment_id, max_iterations=5000):
    """Run a single experiment with the CORRECT hard arguments"""

    # Build command with the EXACT hard setup from the user
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
        "[-2.0,2.0]",  # HARD: includes negatives
        "--extrapolation-range",
        "[[-6.0,-2.0],[2.0,6.0]]",  # HARD: much wider range
        "--no-cuda",
        "--max-iterations",
        str(max_iterations),
        "--log-interval",
        "500",
        "--no-open-browser",
        "--note",
        f"corrected_search_{experiment_id}",
    ]

    # Add hyperparameters to command
    for param, value in config.items():
        if value is not None:
            cmd.extend([f'--{param.replace("_", "-")}', str(value)])

    print(f"Running experiment {experiment_id}: {config}")

    start_time = time.time()

    try:
        # Run the experiment
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout for 5000 iterations
            cwd="/Users/paul_curry/ai2/nalm-benchmark",
        )

        duration = time.time() - start_time

        # Parse the output to extract final losses
        output_lines = result.stdout.strip().split("\n")
        final_losses = {}
        grokked = False
        early_stopped = False

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
                        if inter_loss < 1e-4:  # Slightly relaxed grokking threshold
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

                        final_losses = {
                            "train": train_loss,
                            "interpolation": inter_loss,
                            "extrapolation": extra_loss,
                        }

                        if inter_loss < 1e-4:  # Slightly relaxed grokking threshold
                            grokked = True
                except (ValueError, IndexError):
                    continue

        # Check if we have valid losses
        if not final_losses:
            final_losses = {
                "train": float("inf"),
                "interpolation": float("inf"),
                "extrapolation": float("inf"),
            }

        success = result.returncode == 0

        return {
            "experiment_id": experiment_id,
            "config": config,
            "success": success,
            "grokked": grokked,
            "early_stopped": early_stopped,
            "duration": duration,
            "final_losses": final_losses,
            "stdout": result.stdout[-2000:] if result.stdout else "",  # Last 2000 chars
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
            "stdout": "",
            "stderr": str(e),
        }


def main():
    parser = argparse.ArgumentParser(
        description="CORRECTED hyperparameter search for division grokking with hard ranges"
    )
    parser.add_argument(
        "--max-experiments",
        type=int,
        default=40,
        help="Maximum number of experiments to run",
    )
    parser.add_argument(
        "--max-parallel", type=int, default=3, help="Maximum parallel processes"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5000,
        help="Maximum iterations per experiment",
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default="div_corrected_search_results.json",
        help="Results file",
    )
    parser.add_argument(
        "--conservative-only",
        action="store_true",
        help="Only run ultra-conservative configurations",
    )

    args = parser.parse_args()

    # Create results directory
    results_dir = Path("hyperparameter_search_results")
    results_dir.mkdir(exist_ok=True)

    # Generate configurations
    if args.conservative_only:
        configs = ULTRA_CONSERVATIVE_CONFIGS[: args.max_experiments]
        print(
            f"Running {len(configs)} ultra-conservative configurations with CORRECT hard args"
        )
    else:
        configs = ULTRA_CONSERVATIVE_CONFIGS + generate_random_configs(
            args.max_experiments - len(ULTRA_CONSERVATIVE_CONFIGS)
        )
        configs = configs[: args.max_experiments]
        print(
            f"Running {len(configs)} configurations ({len(ULTRA_CONSERVATIVE_CONFIGS)} ultra-conservative + {len(configs) - len(ULTRA_CONSERVATIVE_CONFIGS)} random)"
        )

    print("IMPORTANT: Using the correct HARD interpolation/extrapolation ranges:")
    print("  --interpolation-range '[-2.0,2.0]'")
    print("  --extrapolation-range '[[-6.0,-2.0],[2.0,6.0]]'")
    print()

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    results = []
    grokked_configs = []
    good_progress_configs = []

    print(
        f"Starting CORRECTED hyperparameter search with {args.max_parallel} parallel processes..."
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

                # Check if this configuration achieved grokking
                if result["grokked"]:
                    grokked_configs.append(result)
                    print(f"🎉 GROKKING ACHIEVED! Experiment {experiment_id}")
                    print(f"   Config: {result['config']}")
                    print(f"   Final losses: {result['final_losses']}")
                    print(f"   Duration: {result['duration']:.1f}s")
                    print(f"   Early stopped: {result['early_stopped']}")
                elif (
                    result["success"] and result["final_losses"]["interpolation"] < 0.3
                ):
                    # Track good progress (better than 0.3 interpolation loss)
                    good_progress_configs.append(result)
                    print(f"📈 Good progress - Experiment {experiment_id}")
                    print(
                        f"   Interpolation loss: {result['final_losses']['interpolation']:.4f}"
                    )
                    print(f"   Config: {result['config']}")

                # Print progress
                completed = len(results)
                print(
                    f"Progress: {completed}/{len(configs)} ({completed/len(configs)*100:.1f}%)"
                )
                print(
                    f"Grokking success rate: {len(grokked_configs)}/{completed} ({len(grokked_configs)/completed*100:.1f}%)"
                )
                print(
                    f"Good progress rate: {len(good_progress_configs)}/{completed} ({len(good_progress_configs)/completed*100:.1f}%)"
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
                                "note": "This is the CORRECTED hard setup matching user args",
                            },
                            "search_space": CORRECTED_SEARCH_SPACE,
                            "ultra_conservative_configs": ULTRA_CONSERVATIVE_CONFIGS,
                            "results": results,
                            "grokked_configs": grokked_configs,
                            "good_progress_configs": good_progress_configs,
                            "summary": {
                                "total_experiments": len(results),
                                "successful_experiments": sum(
                                    1 for r in results if r["success"]
                                ),
                                "grokking_experiments": len(grokked_configs),
                                "good_progress_experiments": len(good_progress_configs),
                                "grokking_rate": (
                                    len(grokked_configs) / len(results)
                                    if results
                                    else 0
                                ),
                                "good_progress_rate": (
                                    len(good_progress_configs) / len(results)
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
                print(f"Experiment {experiment_id} failed with exception: {e}")
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
                        "stdout": "",
                        "stderr": str(e),
                    }
                )

    # Final summary
    print("\n" + "=" * 60)
    print("CORRECTED HYPERPARAMETER SEARCH RESULTS SUMMARY")
    print("=" * 60)
    print(f"Problem setup: HARD interpolation [-2,2], extrapolation [[-6,-2],[2,6]]")
    print(f"Total experiments: {len(results)}")
    print(f"Successful experiments: {sum(1 for r in results if r['success'])}")
    print(f"Grokking experiments (< 1e-4): {len(grokked_configs)}")
    print(f"Good progress experiments (< 0.3): {len(good_progress_configs)}")
    print(f"Grokking rate: {len(grokked_configs)/len(results)*100:.1f}%")
    print(f"Good progress rate: {len(good_progress_configs)/len(results)*100:.1f}%")

    if grokked_configs:
        print(f"\n🎉 GROKKING CONFIGURATIONS:")
        grokked_configs.sort(key=lambda x: x["final_losses"]["interpolation"])

        for i, result in enumerate(grokked_configs):
            print(f"\nGrokking Config {i+1}:")
            print(
                f"  Interpolation Loss: {result['final_losses']['interpolation']:.2e}"
            )
            print(
                f"  Extrapolation Loss: {result['final_losses']['extrapolation']:.2e}"
            )
            print(f"  Duration: {result['duration']:.1f}s")
            print(f"  Config: {result['config']}")

    if good_progress_configs:
        print(f"\n📈 BEST PROGRESS CONFIGURATIONS:")
        good_progress_configs.sort(key=lambda x: x["final_losses"]["interpolation"])

        for i, result in enumerate(good_progress_configs[:5]):  # Top 5
            print(f"\nGood Progress Config {i+1}:")
            print(
                f"  Interpolation Loss: {result['final_losses']['interpolation']:.4f}"
            )
            print(
                f"  Extrapolation Loss: {result['final_losses']['extrapolation']:.4f}"
            )
            print(f"  Duration: {result['duration']:.1f}s")
            print(f"  Config: {result['config']}")

    if not grokked_configs and not good_progress_configs:
        print("\n❌ No configurations achieved grokking or significant progress")
        print("Best performing configurations by interpolation loss:")

        successful_results = [r for r in results if r["success"]]
        successful_results.sort(key=lambda x: x["final_losses"]["interpolation"])

        for i, result in enumerate(successful_results[:5]):
            print(f"\nBest Config {i+1}:")
            print(
                f"  Interpolation Loss: {result['final_losses']['interpolation']:.4f}"
            )
            print(
                f"  Extrapolation Loss: {result['final_losses']['extrapolation']:.4f}"
            )
            print(f"  Duration: {result['duration']:.1f}s")
            print(f"  Config: {result['config']}")

    # Save final results
    final_results_file = results_dir / args.results_file
    with open(final_results_file, "w") as f:
        json.dump(
            {
                "problem_setup": {
                    "interpolation_range": "[-2.0,2.0]",
                    "extrapolation_range": "[[-6.0,-2.0],[2.0,6.0]]",
                    "note": "This is the CORRECTED hard setup matching user args",
                },
                "search_space": CORRECTED_SEARCH_SPACE,
                "ultra_conservative_configs": ULTRA_CONSERVATIVE_CONFIGS,
                "results": results,
                "grokked_configs": grokked_configs,
                "good_progress_configs": good_progress_configs,
                "summary": {
                    "total_experiments": len(results),
                    "successful_experiments": sum(1 for r in results if r["success"]),
                    "grokking_experiments": len(grokked_configs),
                    "good_progress_experiments": len(good_progress_configs),
                    "grokking_rate": (
                        len(grokked_configs) / len(results) if results else 0
                    ),
                    "good_progress_rate": (
                        len(good_progress_configs) / len(results) if results else 0
                    ),
                },
            },
            f,
            indent=2,
            default=str,
        )

    print(f"\nResults saved to: {final_results_file}")

    return len(grokked_configs) > 0 or len(good_progress_configs) > 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
