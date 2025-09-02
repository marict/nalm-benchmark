#!/usr/bin/env python3
"""
Calculate grokking thresholds using frozen "perfect" weights with small perturbations.
This implements the paper's methodology: W* ± ε where ε = 10e-5.
Uses batched inference on 1M samples to calculate proper MSE thresholds.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(".")

# Disable wandb logging
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

import wandb

wandb.init(mode="disabled")

from stable_nalu.dataset import SimpleFunctionStaticDataset
from stable_nalu.layer import DAGLayer

# Test ranges from comprehensive_table_test.py
TEST_RANGES = [
    ([-2, 2], [[-6, -2], [2, 6]], "sym"),  # symmetric around 0
    ([-0.2, -0.1], [-2, -0.2], "n01"),  # negative small (-0.2 to -0.1)
    ([10, 20], [20, 40], "p20"),  # positive large (10-20)
]

OPERATIONS = ["add", "sub", "mul", "div"]  # Test all operations

# Simple freeze configuration - now we just need op, freeze_G, and freeze_O
FREEZE_CONFIG = {
    "freeze_O": True,
    "freeze_G": True,
}

# Use moderate perturbation for realistic thresholds
PERTURBATION = 1e-4


def create_perfect_dag_layer(operation):
    """Create DAG layer with perfect frozen weights."""
    layer = DAGLayer(
        in_features=2,
        out_features=1,
        dag_depth=1,
        op=operation,
        freeze_O=FREEZE_CONFIG["freeze_O"],
        freeze_G=FREEZE_CONFIG["freeze_G"],
        freeze_input_norm=False,  # Match --no-norm from training
        use_norm=False,  # Disable input norm completely
        G_perturbation=0.0,  # No perturbation for perfect weights
    )
    layer.eval()
    return layer


def create_perturbed_dag_layer(operation, perturbation):
    """Create DAG layer with perturbed weights."""
    layer = DAGLayer(
        in_features=2,
        out_features=1,
        dag_depth=1,
        op=operation,
        freeze_O=FREEZE_CONFIG["freeze_O"],
        freeze_G=FREEZE_CONFIG["freeze_G"],
        freeze_input_norm=False,  # Match --no-norm from training
        use_norm=False,  # Disable input norm completely
        G_perturbation=perturbation,
    )
    # Keep in training mode to use soft G values with perturbation
    layer.train()
    return layer


def generate_test_data(extrap_range, n_samples=1_000_000, batch_size=10000):
    """Generate test data for extrapolation range."""
    if isinstance(extrap_range[0], list):
        # Handle nested ranges like [[-6, -2], [2, 6]]
        all_samples = []
        for sub_range in extrap_range:
            n_sub = n_samples // len(extrap_range)
            x1 = np.random.uniform(sub_range[0], sub_range[1], n_sub)
            x2 = np.random.uniform(sub_range[0], sub_range[1], n_sub)
            all_samples.append(np.column_stack([x1, x2]))
        samples = np.vstack(all_samples)
    else:
        # Simple range
        x1 = np.random.uniform(extrap_range[0], extrap_range[1], n_samples)
        x2 = np.random.uniform(extrap_range[0], extrap_range[1], n_samples)
        samples = np.column_stack([x1, x2])

    return torch.tensor(samples, dtype=torch.float32)


def calculate_threshold_mse(operation, interp_range, extrap_range, range_name):
    """Calculate threshold MSE using paper's W* ± ε methodology on 1M samples."""

    try:
        # Create perfect and perturbed models
        perfect_layer = create_perfect_dag_layer(operation)
        perturbed_layer = create_perturbed_dag_layer(operation, PERTURBATION)

        # Generate test data
        print(f"    Generating 1M test samples...", end="", flush=True)
        test_data = generate_test_data(extrap_range)

        # Calculate true targets
        if operation == "add":
            targets = test_data[:, 0] + test_data[:, 1]
        elif operation == "sub":
            targets = test_data[:, 0] - test_data[:, 1]
        elif operation == "mul":
            targets = test_data[:, 0] * test_data[:, 1]
        elif operation == "div":
            targets = test_data[:, 0] / test_data[:, 1]

        targets = targets.unsqueeze(1)

        # Batched inference to avoid memory issues
        batch_size = 10000
        perfect_mse = 0.0
        perturbed_mse = 0.0

        print(f" Running batched inference...", end="", flush=True)

        with torch.no_grad():
            for i in range(0, len(test_data), batch_size):
                batch_data = test_data[i : i + batch_size]
                batch_targets = targets[i : i + batch_size]

                # Perfect model predictions
                perfect_pred = perfect_layer(batch_data)
                perfect_batch_mse = torch.mean((perfect_pred - batch_targets) ** 2)
                perfect_mse += perfect_batch_mse.item() * len(batch_data)

                # Perturbed model predictions
                perturbed_pred = perturbed_layer(batch_data)
                perturbed_batch_mse = torch.mean((perturbed_pred - batch_targets) ** 2)
                perturbed_mse += perturbed_batch_mse.item() * len(batch_data)

        # Calculate final MSE
        perfect_mse /= len(test_data)
        perturbed_mse /= len(test_data)

        return {
            "perfect_mse": perfect_mse,
            "perturbed_mse": perturbed_mse,
            "threshold": perturbed_mse,  # This is the threshold for grokking
        }

    except Exception as e:
        print(f"\n    ERROR: {e}")
        return None


def main():
    print("CALCULATING GROKKING THRESHOLDS")
    print("=" * 60)
    print(f"Using paper methodology: W* ± ε where ε = {PERTURBATION}")
    print("Batched inference on 1M samples per test")
    print(
        f"\nUsing simplified freezing: freeze_O={FREEZE_CONFIG['freeze_O']}, freeze_G={FREEZE_CONFIG['freeze_G']}"
    )
    print("Operation-specific patterns determined automatically by op parameter")
    print()

    results = {}

    for operation in OPERATIONS:
        print(f"\n🔧 Testing {operation.upper()} operation:")
        results[operation] = {}

        for interp_range, extrap_range, range_name in TEST_RANGES:
            print(f"  {range_name}: ", end="", flush=True)

            threshold_data = calculate_threshold_mse(
                operation, interp_range, extrap_range, range_name
            )

            if threshold_data is not None:
                results[operation][range_name] = threshold_data
                print(
                    f" Perfect: {threshold_data['perfect_mse']:.2e}, Threshold: {threshold_data['threshold']:.2e}"
                )
            else:
                results[operation][range_name] = None
                print(" FAILED")

    # Save results
    output_dir = Path("experiment_results")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "grokking_thresholds_paper_method.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n📊 THRESHOLD SUMMARY:")
    print("=" * 80)

    # Print summary table
    print(
        f"{'Operation':<10} {'Range':<8} {'Perfect MSE':<12} {'Threshold MSE':<12} {'Ratio':<10}"
    )
    print("-" * 80)

    for operation in OPERATIONS:
        for range_name in [r[2] for r in TEST_RANGES]:
            threshold_data = results[operation].get(range_name)
            if threshold_data is not None:
                perfect_mse = threshold_data["perfect_mse"]
                threshold_mse = threshold_data["threshold"]
                ratio = threshold_mse / perfect_mse if perfect_mse > 0 else float("inf")
                print(
                    f"{operation.upper():<10} {range_name:<8} {perfect_mse:<12.2e} {threshold_mse:<12.2e} {ratio:<10.1e}"
                )
            else:
                print(
                    f"{operation.upper():<10} {range_name:<8} {'FAILED':<12} {'FAILED':<12} {'N/A':<10}"
                )

    print(f"\nResults saved to: {output_file}")
    print(f"\nThese thresholds represent the MSE that models must achieve")
    print(f"to be considered as having 'grokked' the operation on each range.")


if __name__ == "__main__":
    main()
