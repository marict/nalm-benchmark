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

# More aggressive hyperparameter search for division grokking within 2000 iterations
# Based on gradient analysis showing 1e+12 gradient norms, we need VERY aggressive clipping
# and much lower learning rates

AGGRESSIVE_SEARCH_SPACE = {
    # Much lower learning rates for stability
    "learning_rate": [1e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4],
    # Very aggressive gradient clipping
    "clip_grad_norm": [0.0001, 0.001, 0.01, 0.1],
    # High regularization to prevent explosions
    "regualizer": [100.0, 500.0, 1000.0],
    "regualizer_z": [10.0, 50.0, 100.0],
    "regualizer_oob": [10.0, 50.0, 100.0],
    # Simpler architectures
    "hidden_size": [1, 2],
    "batch_size": [32, 64, 128],
    # Only Adam (better gradient handling)
    "optimizer": ["adam"],
    # Minimal DAG complexity
    "num_subsets": [1, 2],
    # Strong target filtering
    "max_target_magnitude": [1.0, 5.0, 10.0],
    # Strong division regularizer
    "div_regularizer": [1e-3, 1e-2, 1e-1],
}

# Ultra-conservative configurations
ULTRA_CONSERVATIVE_CONFIGS = [
    {
        "learning_rate": 1e-6,
        "clip_grad_norm": 0.0001,
        "regualizer": 1000.0,
        "regualizer_z": 100.0,
        "regualizer_oob": 100.0,
        "hidden_size": 1,
        "batch_size": 32,
        "optimizer": "adam",
        "num_subsets": 1,
        "max_target_magnitude": 1.0,
        "div_regularizer": 1e-1,
    },
    {
        "learning_rate": 5e-6,
        "clip_grad_norm": 0.001,
        "regualizer": 500.0,
        "regualizer_z": 50.0,
        "regualizer_oob": 50.0,
        "hidden_size": 1,
        "batch_size": 64,
        "optimizer": "adam",
        "num_subsets": 1,
        "max_target_magnitude": 5.0,
        "div_regularizer": 1e-2,
    },
    {
        "learning_rate": 1e-5,
        "clip_grad_norm": 0.01,
        "regualizer": 100.0,
        "regualizer_z": 10.0,
        "regualizer_oob": 10.0,
        "hidden_size": 2,
        "batch_size": 128,
        "optimizer": "adam",
        "num_subsets": 2,
        "max_target_magnitude": 10.0,
        "div_regularizer": 1e-3,
    },
    # Try with cosine LR schedule
    {
        "learning_rate": 1e-4,
        "clip_grad_norm": 0.1,
        "regualizer": 100.0,
        "regualizer_z": 10.0,
        "regualizer_oob": 10.0,
        "hidden_size": 1,
        "batch_size": 64,
        "optimizer": "adam",
        "num_subsets": 1,
        "max_target_magnitude": 5.0,
        "div_regularizer": 1e-2,
        "lr_cosine": True,
        "lr_min": 1e-7,
    },
    # Different layer type - try basic NALU
    {
        "learning_rate": 1e-4,
        "clip_grad_norm": 0.01,
        "regualizer": 10.0,
        "regualizer_z": 1.0,
        "regualizer_oob": 1.0,
        "hidden_size": 2,
        "batch_size": 128,
        "optimizer": "adam",
        "max_target_magnitude": 10.0,
        "layer_type": "NALU",
    },
]


def generate_aggressive_configs(n_configs=50):
    """Generate aggressive hyperparameter configurations"""
    configs = []

    for _ in range(n_configs):
        config = {}
        for param, values in AGGRESSIVE_SEARCH_SPACE.items():
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
        "--max-iterations",
        str(max_iterations),
        "--seed",
        str(random.randint(0, 9999)),
        "--log-interval",
        "200",  # Even more frequent logging
        "--no-open-browser",
        "--note",
        f"aggressive_{experiment_id}",
    ]

    # Default layer type
    layer_type = config.get("layer_type", "DAG")
    cmd.extend(["--layer-type", layer_type])

    # Add hyperparameters to command
    for param, value in config.items():
        if param == "layer_type":
            continue  # Already added

        if param == "lr_cosine" and value:
            cmd.append("--lr-cosine")
            continue

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
            timeout=400,  # 6.67 minute timeout for 2000 iterations
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
                    # Extract step number and losses
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
                        if inter_loss < 1e-5:
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

                        if inter_loss < 1e-5:
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
        description="Aggressive hyperparameter search for division grokking"
    )
    parser.add_argument(
        "--max-experiments",
        type=int,
        default=30,
        help="Maximum number of experiments to run",
    )
    parser.add_argument(
        "--max-parallel", type=int, default=3, help="Maximum parallel processes"
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
        default="div_aggressive_search_results.json",
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
        print(f"Running {len(configs)} ultra-conservative configurations")
    else:
        configs = ULTRA_CONSERVATIVE_CONFIGS + generate_aggressive_configs(
            args.max_experiments - len(ULTRA_CONSERVATIVE_CONFIGS)
        )
        configs = configs[: args.max_experiments]
        print(
            f"Running {len(configs)} configurations ({len(ULTRA_CONSERVATIVE_CONFIGS)} ultra-conservative + {len(configs) - len(ULTRA_CONSERVATIVE_CONFIGS)} aggressive)"
        )

    # Set random seed for reproducibility
    random.seed(123)
    np.random.seed(123)

    results = []
    grokked_configs = []

    print(
        f"Starting aggressive hyperparameter search with {args.max_parallel} parallel processes..."
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
                else:
                    # Print best performing non-grokking results
                    if (
                        result["success"]
                        and result["final_losses"]["interpolation"] < 1.0
                    ):
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
                    f"Grokking success rate so far: {len(grokked_configs)}/{completed} ({len(grokked_configs)/completed*100:.1f}%)"
                )
                print()

                # Save intermediate results
                results_file = results_dir / args.results_file
                with open(results_file, "w") as f:
                    json.dump(
                        {
                            "search_space": AGGRESSIVE_SEARCH_SPACE,
                            "ultra_conservative_configs": ULTRA_CONSERVATIVE_CONFIGS,
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
    print("AGGRESSIVE HYPERPARAMETER SEARCH RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total experiments: {len(results)}")
    print(f"Successful experiments: {sum(1 for r in results if r['success'])}")
    print(f"Grokking experiments: {len(grokked_configs)}")
    print(f"Grokking rate: {len(grokked_configs)/len(results)*100:.1f}%")

    if grokked_configs:
        print(f"\n🎉 BEST GROKKING CONFIGURATIONS:")
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

        successful_results = [r for r in results if r["success"]]
        successful_results.sort(key=lambda x: x["final_losses"]["interpolation"])

        for i, result in enumerate(successful_results[:5]):
            print(f"\nRank {i+1}:")
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
                "search_space": AGGRESSIVE_SEARCH_SPACE,
                "ultra_conservative_configs": ULTRA_CONSERVATIVE_CONFIGS,
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
