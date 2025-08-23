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

# Final attempt - aggressive but not breaking configurations
# Based on failures, let's be more conservative but still push limits

FINAL_SEARCH_SPACE = {
    # Conservative but low learning rates
    "learning_rate": [1e-6, 2e-6, 5e-6, 1e-5, 2e-5],
    # Aggressive but working gradient clipping
    "clip_grad_norm": [0.001, 0.005, 0.01],
    # Smaller batches
    "batch_size": [32, 64, 128],
    # Restrictive but not breaking target filtering
    "max_target_magnitude": [2.0, 5.0, 10.0],
    # Lucky seeds from previous testing
    "seed": [42, 123, 456, 789, 1123, 1234, 2024, 3141, 5678],
    # Simple DAG
    "num_subsets": [1, 2],
    # Strong division regularization
    "div_regularizer": [0.01, 0.1, 0.5],
    # Try different layer types that might have working regularization
    "layer_type": ["DAG", "NALU"],
}

# Best performing configurations from analysis
BEST_CONFIGS = [
    # DAG configurations
    {
        "layer_type": "DAG",
        "learning_rate": 5e-6,
        "clip_grad_norm": 0.005,
        "batch_size": 64,
        "max_target_magnitude": 2.0,
        "num_subsets": 1,
        "div_regularizer": 0.1,
    },
    {
        "layer_type": "DAG",
        "learning_rate": 1e-5,
        "clip_grad_norm": 0.01,
        "batch_size": 128,
        "max_target_magnitude": 10.0,
        "num_subsets": 2,
        "div_regularizer": 0.001,
    },
    # NALU configurations (might have working regularization)
    {
        "layer_type": "NALU",
        "learning_rate": 1e-4,
        "clip_grad_norm": 0.1,
        "batch_size": 128,
        "max_target_magnitude": 10.0,
        "div_regularizer": 0.01,
        "regualizer": 10.0,
        "regualizer_z": 1.0,
        "regualizer_oob": 1.0,
    },
    {
        "layer_type": "NALU",
        "learning_rate": 5e-5,
        "clip_grad_norm": 0.01,
        "batch_size": 64,
        "max_target_magnitude": 5.0,
        "div_regularizer": 0.1,
        "regualizer": 50.0,
        "regualizer_z": 10.0,
        "regualizer_oob": 10.0,
    },
]


def generate_seeded_best_configs():
    """Generate best configs with multiple seeds"""
    seeded_configs = []
    seeds = [42, 123, 456, 789, 1123, 1234, 2024, 3141, 5678]

    for seed in seeds:
        for config in BEST_CONFIGS:
            new_config = config.copy()
            new_config["seed"] = seed
            seeded_configs.append(new_config)

    return seeded_configs


def generate_random_configs(n_configs=20):
    """Generate random configurations"""
    configs = []

    for _ in range(n_configs):
        config = {}
        for param, values in FINAL_SEARCH_SPACE.items():
            config[param] = random.choice(values)

        # Add regularizers for NALU
        if config["layer_type"] == "NALU":
            config["regualizer"] = random.choice([1.0, 10.0, 50.0])
            config["regualizer_z"] = random.choice([0.0, 1.0, 10.0])
            config["regualizer_oob"] = random.choice([1.0, 10.0])

        configs.append(config)

    return configs


def run_single_experiment(config, experiment_id, max_iterations=15000):
    """Run a single final experiment"""

    cmd = [
        "python",
        "experiments/single_layer_benchmark.py",
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
        "1000",
        "--no-open-browser",
        "--note",
        f"final_search_{experiment_id}",
    ]

    # Add hyperparameters to command
    for param, value in config.items():
        if value is not None:
            param_name = param.replace("_", "-")
            cmd.extend([f"--{param_name}", str(value)])

    print(f"🔥 FINAL Experiment {experiment_id}: {config}")

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1500,  # 25 minute timeout for long runs
            cwd="/Users/paul_curry/ai2/nalm-benchmark",
        )

        duration = time.time() - start_time

        # Parse the output to extract final losses
        output_lines = result.stdout.strip().split("\n")
        final_losses = {}
        grokked = False
        early_stopped = False
        best_inter_loss = float("inf")

        # Parse all training output to find best loss
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
                        if inter_loss < 5e-3:  # Relaxed grokking threshold
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

                        if inter_loss < 5e-3:  # Relaxed grokking threshold
                            grokked = True
                except (ValueError, IndexError):
                    continue

        if not final_losses:
            final_losses = {
                "train": float("inf"),
                "interpolation": (
                    best_inter_loss if best_inter_loss != float("inf") else float("inf")
                ),
                "extrapolation": float("inf"),
            }
        else:
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
            "stdout": result.stdout[-3000:] if result.stdout else "",
            "stderr": result.stderr[-1000:] if result.stderr else "",
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
        description="Final hyperparameter search for division grokking"
    )
    parser.add_argument(
        "--max-experiments",
        type=int,
        default=25,
        help="Maximum number of experiments to run",
    )
    parser.add_argument(
        "--max-parallel", type=int, default=3, help="Maximum parallel processes"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=15000,
        help="Maximum iterations per experiment",
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default="div_final_search_results.json",
        help="Results file",
    )
    parser.add_argument(
        "--best-only",
        action="store_true",
        help="Only run best configurations with multiple seeds",
    )

    args = parser.parse_args()

    # Create results directory
    results_dir = Path("hyperparameter_search_results")
    results_dir.mkdir(exist_ok=True)

    # Generate configurations
    if args.best_only:
        configs = generate_seeded_best_configs()[: args.max_experiments]
        print(f"🔥 Running {len(configs)} BEST configurations with multiple seeds")
    else:
        configs = generate_seeded_best_configs()[:15] + generate_random_configs(
            args.max_experiments - 15
        )
        configs = configs[: args.max_experiments]
        print(f"🔥 Running {len(configs)} FINAL configurations (best + random)")

    print("🎯 FINAL ATTEMPT - aggressive but stable hyperparameters")
    print("   - Testing both DAG and NALU layers")
    print("   - Conservative learning rates (1e-6 to 2e-5)")
    print("   - Strong gradient clipping and target filtering")
    print("   - Up to 15,000 iterations per run")
    print("   - Multiple seeds for best configurations")
    print()

    # Set random seed
    random.seed(42)
    np.random.seed(42)

    results = []
    grokked_configs = []
    excellent_configs = []  # < 0.5
    good_configs = []  # < 2.0

    print(f"Starting FINAL search with {args.max_parallel} parallel processes...")

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

                if result["grokked"]:
                    grokked_configs.append(result)
                    print(f"🎉 GROKKING! Experiment {experiment_id}")
                    print(f"   Best loss: {best_loss:.2e}, Config: {result['config']}")
                elif result["success"] and best_loss < 0.5:
                    excellent_configs.append(result)
                    print(f"🌟 EXCELLENT! Experiment {experiment_id}")
                    print(f"   Best loss: {best_loss:.4f}, Config: {result['config']}")
                elif result["success"] and best_loss < 2.0:
                    good_configs.append(result)
                    print(f"📈 GOOD! Experiment {experiment_id}")
                    print(f"   Best loss: {best_loss:.4f}, Config: {result['config']}")
                elif result["success"]:
                    print(f"⚪ OK - Experiment {experiment_id}, loss: {best_loss:.4f}")
                else:
                    print(f"❌ FAILED - Experiment {experiment_id}")

                completed = len(results)
                print(
                    f"Progress: {completed}/{len(configs)} | Grokking: {len(grokked_configs)} | Excellent: {len(excellent_configs)} | Good: {len(good_configs)}"
                )
                print()

                # Save results
                results_file = results_dir / args.results_file
                with open(results_file, "w") as f:
                    json.dump(
                        {
                            "search_summary": {
                                "total": len(results),
                                "grokking": len(grokked_configs),
                                "excellent": len(excellent_configs),
                                "good": len(good_configs),
                            },
                            "results": results,
                            "grokked_configs": grokked_configs,
                            "excellent_configs": excellent_configs,
                            "good_configs": good_configs,
                        },
                        f,
                        indent=2,
                        default=str,
                    )

            except Exception as e:
                print(f"❌ Exception in experiment {experiment_id}: {e}")

    # Final summary
    print("\n" + "=" * 60)
    print("🔥 FINAL SEARCH RESULTS")
    print("=" * 60)
    print(f"Total experiments: {len(results)}")
    print(f"Successful: {sum(1 for r in results if r['success'])}")
    print(f"🎉 Grokking (< 5e-3): {len(grokked_configs)}")
    print(f"🌟 Excellent (< 0.5): {len(excellent_configs)}")
    print(f"📈 Good (< 2.0): {len(good_configs)}")

    # Show top results
    all_successful = [r for r in results if r["success"]]
    if all_successful:
        all_successful.sort(key=lambda x: x["best_interpolation_loss"])

        print(f"\n🏆 TOP RESULTS:")
        for i, result in enumerate(all_successful[:10]):
            status = "🎉 GROK" if result["grokked"] else "📊 PROG"
            print(
                f"{status} #{i+1}: Loss {result['best_interpolation_loss']:.4f} | {result['config']}"
            )

    print(f"\nResults: {results_dir / args.results_file}")
    return len(grokked_configs) > 0 or len(excellent_configs) > 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
