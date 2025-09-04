#!/usr/bin/env python3
"""
Minimal test to verify G perturbation is working.
"""

import sys

import torch

sys.path.append(".")

from stable_nalu.layer import DAGLayer


def test_perturbation():
    print("Testing G perturbation implementation...")

    # Create test input
    test_input = torch.tensor([[2.0, 3.0]], dtype=torch.float32)

    # Test ADD operation (should prefer linear domain)
    print("\n=== ADD OPERATION ===")

    # Perfect layer (no perturbation)
    perfect_layer = DAGLayer(
        in_features=2,
        out_features=1,
        op="add",
        freeze_O=True,
        freeze_G=True,
        use_norm=False,
        G_perturbation=0.0,
        single_G=True,
    )
    perfect_layer.eval()

    # Perturbed layer
    perturbed_layer = DAGLayer(
        in_features=2,
        out_features=1,
        op="add",
        freeze_O=True,
        freeze_G=True,
        use_norm=False,
        G_perturbation=1e-4,
        single_G=True,
    )
    perturbed_layer.eval()

    with torch.no_grad():
        perfect_pred = perfect_layer(test_input)
        perturbed_pred = perturbed_layer(test_input)

        print(f"Input: {test_input}")
        print(f"Perfect prediction: {perfect_pred.item():.8f}")
        print(f"Perturbed prediction: {perturbed_pred.item():.8f}")
        print(f"True result (2+3): {5.0}")
        print(f"Difference: {abs(perfect_pred.item() - perturbed_pred.item()):.2e}")

    # Test MUL operation (should prefer log domain)
    print("\n=== MUL OPERATION ===")

    # Perfect layer
    perfect_mul = DAGLayer(
        in_features=2,
        out_features=1,
        op="mul",
        freeze_O=True,
        freeze_G=True,
        use_norm=False,
        G_perturbation=0.0,
        single_G=True,
    )
    perfect_mul.eval()

    # Perturbed layer
    perturbed_mul = DAGLayer(
        in_features=2,
        out_features=1,
        op="mul",
        freeze_O=True,
        freeze_G=True,
        use_norm=False,
        G_perturbation=1e-4,
        single_G=True,
    )
    perturbed_mul.eval()

    with torch.no_grad():
        perfect_mul_pred = perfect_mul(test_input)
        perturbed_mul_pred = perturbed_mul(test_input)

        print(f"Input: {test_input}")
        print(f"Perfect prediction: {perfect_mul_pred.item():.8f}")
        print(f"Perturbed prediction: {perturbed_mul_pred.item():.8f}")
        print(f"True result (2*3): {6.0}")
        print(
            f"Difference: {abs(perfect_mul_pred.item() - perturbed_mul_pred.item()):.2e}"
        )


if __name__ == "__main__":
    test_perturbation()
