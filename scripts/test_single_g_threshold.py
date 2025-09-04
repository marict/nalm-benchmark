#!/usr/bin/env python3
"""
Quick test of single_G threshold calculation for one operation and range.
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

from stable_nalu.layer import DAGLayer

# Test just one operation and one range
OPERATION = "add"
RANGE_NAME = "sym"
INTERP_RANGE = [-2, 2]
EXTRAP_RANGE = [[-6, -2], [2, 6]]
PERTURBATION = 1e-4


def create_perfect_dag_layer(operation, single_G=True):
    """Create DAG layer with perfect frozen weights."""
    layer = DAGLayer(
        in_features=2,
        out_features=1,
        dag_depth=1,
        op=operation,
        freeze_O=True,
        freeze_G=True,
        freeze_input_norm=False,
        use_norm=False,
        G_perturbation=0.0,
        single_G=single_G,
    )
    layer.eval()
    return layer


def create_perturbed_dag_layer(operation, perturbation, single_G=True):
    """Create DAG layer with perturbed weights."""
    layer = DAGLayer(
        in_features=2,
        out_features=1,
        dag_depth=1,
        op=operation,
        freeze_O=True,
        freeze_G=True,
        freeze_input_norm=False,
        use_norm=False,
        G_perturbation=perturbation,
        single_G=single_G,
    )
    layer.train()
    return layer


def generate_test_data(extrap_range, n_samples=10000):
    """Generate test data for extrapolation range."""
    if isinstance(extrap_range[0], list):
        all_samples = []
        for sub_range in extrap_range:
            n_sub = n_samples // len(extrap_range)
            x1 = np.random.uniform(sub_range[0], sub_range[1], n_sub)
            x2 = np.random.uniform(sub_range[0], sub_range[1], n_sub)
            all_samples.append(np.column_stack([x1, x2]))
        samples = np.vstack(all_samples)
    else:
        x1 = np.random.uniform(extrap_range[0], extrap_range[1], n_samples)
        x2 = np.random.uniform(extrap_range[0], extrap_range[1], n_samples)
        samples = np.column_stack([x1, x2])

    return torch.tensor(samples, dtype=torch.float32)


def main():
    print(f"Testing single_G threshold for {OPERATION} operation on {RANGE_NAME} range")
    print(f"Using perturbation = {PERTURBATION}")

    try:
        # Create models
        print("Creating perfect model...", end="")
        perfect_layer = create_perfect_dag_layer(OPERATION, single_G=True)
        print(" ✓")

        print("Creating perturbed model...", end="")
        perturbed_layer = create_perturbed_dag_layer(
            OPERATION, PERTURBATION, single_G=True
        )
        print(" ✓")

        # Generate test data
        print("Generating test data...", end="")
        test_data = generate_test_data(EXTRAP_RANGE)
        print(f" {len(test_data)} samples ✓")

        # Calculate targets
        print("Calculating targets...", end="")
        if OPERATION == "add":
            targets = test_data[:, 0] + test_data[:, 1]
        elif OPERATION == "sub":
            targets = test_data[:, 0] - test_data[:, 1]
        elif OPERATION == "mul":
            targets = test_data[:, 0] * test_data[:, 1]
        elif OPERATION == "div":
            targets = test_data[:, 0] / test_data[:, 1]
        targets = targets.unsqueeze(1)
        print(" ✓")

        # Run inference
        print("Running inference...", end="")
        with torch.no_grad():
            perfect_pred = perfect_layer(test_data)
            perfect_mse = torch.mean((perfect_pred - targets) ** 2).item()

            perturbed_pred = perturbed_layer(test_data)
            perturbed_mse = torch.mean((perturbed_pred - targets) ** 2).item()
        print(" ✓")

        # Results
        print(f"\nResults for {OPERATION} {RANGE_NAME}:")
        print(f"  Perfect MSE:   {perfect_mse:.2e}")
        print(f"  Perturbed MSE: {perturbed_mse:.2e}")
        print(
            f"  Ratio:         {perturbed_mse/perfect_mse:.1e}"
            if perfect_mse > 0
            else "  Ratio: inf"
        )

        return {
            "perfect_mse": perfect_mse,
            "perturbed_mse": perturbed_mse,
            "threshold": perturbed_mse,
        }

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = main()
    if result:
        print("\n✅ Test completed successfully!")
    else:
        print("\n❌ Test failed!")
