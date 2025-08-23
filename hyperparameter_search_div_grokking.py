#!/usr/bin/env python3

import argparse
import itertools
import json
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# Hyperparameter search for division grokking within 2000 iterations
# Based on the gradient analysis showing extreme instability, focus on:
# 1. Gradient clipping (essential given 1e+12 gradient norms)
# 2. Learning rate (lower rates for stability)
# 3. Regularization (to prevent explosive growth)
# 4. Architecture parameters (hidden size, DAG depth)

SEARCH_SPACE = {
    # Critical stability parameters
    "learning_rate": [1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3],
    "clip_grad_norm": [0.001, 0.01, 0.1, 1.0, 10.0],
    # Regularization to control gradient explosions
    "regualizer": [1.0, 5.0, 10.0, 50.0, 100.0],
    "regualizer_z": [0.0, 1.0, 10.0],
    "regualizer_oob": [0.1, 1.0, 10.0],
    # Architecture parameters
    "hidden_size": [1, 2, 4],
    "batch_size": [32, 64, 128, 256],
    # Optimizer choice
    "optimizer": ["adam", "sgd"],
    "momentum": [0.0, 0.9],  # Only for SGD
    # DAG-specific parameters
    "num_subsets": [1, 2, 3],
    # Target filtering to avoid extreme division results
    "max_target_magnitude": [10.0, 100.0, 1000.0],
    # Division regularizer for numerical stability
    "div_regularizer": [None, 1e-6, 1e-4, 1e-2],
}

# Quick grokking configurations to try first
PRIORITY_CONFIGS = [
    {
        "learning_rate": 1e-4,
        "clip_grad_norm": 0.01,
        "regualizer": 50.0,
        "regualizer_z": 10.0,
        "regualizer_oob": 10.0,
        "hidden_size": 2,
        "batch_size": 128,
        "optimizer": "adam",
        "momentum": 0.0,
        "num_subsets": 2,
        "max_target_magnitude": 100.0,
        "div_regularizer": 1e-4,
    },
    {
        "learning_rate": 5e-5,
        "clip_grad_norm": 0.1,
        "regualizer": 100.0,
        "regualizer_z": 1.0,
        "regualizer_oob": 1.0,
        "hidden_size": 1,
        "batch_size": 64,
        "optimizer": "adam",
        "momentum": 0.0,
        "num_subsets": 1,
        "max_target_magnitude": 10.0,
        "div_regularizer": 1e-6,
    },
    {
        "learning_rate": 2e-4,
        "clip_grad_norm": 1.0,
        "regualizer": 10.0,
        "regualizer_z": 0.0,
        "regualizer_oob": 0.1,
        "hidden_size": 4,
        "batch_size": 256,
        "optimizer": "sgd",
        "momentum": 0.9,
        "num_subsets": 3,
        "max_target_magnitude": 1000.0,
        "div_regularizer": None,
    },
]


def generate_random_configs(n_configs=100):
    """Generate random hyperparameter configurations"""
    configs = []

    for _ in range(n_configs):
        config = {}
        for param, values in SEARCH_SPACE.items():
            config[param] = random.choice(values)
        configs.append(config)

    return configs


def run_single_experiment(config, experiment_id, max_iterations=2000):
    """Run a single experiment with given hyperparameters"""

    # Build command
    cmd = [
        "python",
        "experiments/single_layer_benchmark.py",
        "--operation",
        "div",
        "--layer-type",
        "DAG",
        "--max-iterations",
        str(max_iterations),
        "--seed",
        str(random.randint(0, 9999)),
        "--log-interval",
        "500",  # More frequent logging for short runs
        "--no-open-browser",  # Don't open browser for each run
        "--note",
        f"hypersearch_{experiment_id}",
    ]

    # Add hyperparameters to command
    for param, value in config.items():
        if param == "momentum" and config.get("optimizer") != "sgd":
            continue  # Skip momentum for non-SGD optimizers

        if value is not None:
            cmd.extend([f'--{param.replace("_", "-")}', str(value)])

    print(f"Running experiment {experiment_id}: {config}")
    print(f"Command: {' '.join(cmd)}")

    start_time = time.time()

    try:
        # Run the experiment
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout for 2000 iterations
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
                # Extract losses from early stopping message
                parts = line.split(":")
                if len(parts) >= 2:
                    loss_info = parts[1].strip()
                    if "inter=" in loss_info and "extra=" in loss_info:
                        inter_loss = float(loss_info.split("inter=")[1].split(",")[0])
                        extra_loss = float(loss_info.split("extra=")[1])
                        final_losses = {
                            "train": inter_loss,
                            "interpolation": inter_loss,
                            "extrapolation": extra_loss,
                        }
                        if inter_loss < 1e-5:  # Grokking threshold
                            grokked = True
                        break
            elif line.startswith("train ") and ":" in line:
                # Parse training line: "train 2000: 0.1234567890, inter: 0.1234567890, extra: 0.1234567890"
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

                        if inter_loss < 1e-5:  # Grokking threshold
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
        description="Hyperparameter search for division grokking"
    )
    parser.add_argument(
        "--max-experiments",
        type=int,
        default=50,
        help="Maximum number of experiments to run",
    )
    parser.add_argument(
        "--max-parallel", type=int, default=4, help="Maximum parallel processes"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=2000,
        help="Maximum iterations per experiment",
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default="div_grokking_search_results.json",
        help="Results file",
    )
    parser.add_argument(
        "--priority-only", action="store_true", help="Only run priority configurations"
    )

    args = parser.parse_args()

    # Create results directory
    results_dir = Path("hyperparameter_search_results")
    results_dir.mkdir(exist_ok=True)

    # Generate configurations
    if args.priority_only:
        configs = PRIORITY_CONFIGS[: args.max_experiments]
        print(f"Running {len(configs)} priority configurations")
    else:
        configs = PRIORITY_CONFIGS + generate_random_configs(
            args.max_experiments - len(PRIORITY_CONFIGS)
        )
        configs = configs[: args.max_experiments]
        print(
            f"Running {len(configs)} configurations ({len(PRIORITY_CONFIGS)} priority + {len(configs) - len(PRIORITY_CONFIGS)} random)"
        )

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    results = []
    grokked_configs = []

    print(
        f"Starting hyperparameter search with {args.max_parallel} parallel processes..."
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

                # Print progress
                completed = len(results)
                print(
                    f"Progress: {completed}/{len(configs)} ({completed/len(configs)*100:.1f}%)"
                )
                print(
                    f"Grokking success rate so far: {len(grokked_configs)}/{completed} ({len(grokked_configs)/completed*100:.1f}%)"
                )

                # Save intermediate results
                results_file = results_dir / args.results_file
                with open(results_file, "w") as f:
                    json.dump(
                        {
                            "search_space": SEARCH_SPACE,
                            "results": results,
                            "grokked_configs": grokked_configs,
                            "summary": {
                                "total_experiments": len(results),
                                "successful_experiments": sum(
                                    1 for r in results if r["success"]
                                ),
                                "grokking_experiments": len(grokked_configs),
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
    print("HYPERPARAMETER SEARCH RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total experiments: {len(results)}")
    print(f"Successful experiments: {sum(1 for r in results if r['success'])}")
    print(f"Grokking experiments: {len(grokked_configs)}")
    print(f"Grokking rate: {len(grokked_configs)/len(results)*100:.1f}%")

    if grokked_configs:
        print(f"\n🎉 BEST GROKKING CONFIGURATIONS:")
        # Sort by interpolation loss
        grokked_configs.sort(key=lambda x: x["final_losses"]["interpolation"])

        for i, result in enumerate(grokked_configs[:5]):  # Top 5
            print(f"\nRank {i+1}:")
            print(
                f"  Interpolation Loss: {result['final_losses']['interpolation']:.2e}"
            )
            print(
                f"  Extrapolation Loss: {result['final_losses']['extrapolation']:.2e}"
            )
            print(f"  Duration: {result['duration']:.1f}s")
            print(f"  Early stopped: {result['early_stopped']}")
            print(f"  Config: {result['config']}")
    else:
        print("\n❌ No configurations achieved grokking within 2000 iterations")
        print("Best performing configurations by interpolation loss:")

        # Sort by interpolation loss
        results.sort(key=lambda x: x["final_losses"]["interpolation"])

        for i, result in enumerate(results[:5]):  # Top 5
            if result["success"]:
                print(f"\nRank {i+1}:")
                print(
                    f"  Interpolation Loss: {result['final_losses']['interpolation']:.2e}"
                )
                print(
                    f"  Extrapolation Loss: {result['final_losses']['extrapolation']:.2e}"
                )
                print(f"  Duration: {result['duration']:.1f}s")
                print(f"  Config: {result['config']}")

    # Save final results
    final_results_file = results_dir / args.results_file
    with open(final_results_file, "w") as f:
        json.dump(
            {
                "search_space": SEARCH_SPACE,
                "results": results,
                "grokked_configs": grokked_configs,
                "summary": {
                    "total_experiments": len(results),
                    "successful_experiments": sum(1 for r in results if r["success"]),
                    "grokking_experiments": len(grokked_configs),
                    "grokking_rate": (
                        len(grokked_configs) / len(results) if results else 0
                    ),
                },
            },
            f,
            indent=2,
            default=str,
        )

    print(f"\nResults saved to: {final_results_file}")

    return len(grokked_configs) > 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
