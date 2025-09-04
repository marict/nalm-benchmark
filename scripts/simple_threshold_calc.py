#!/usr/bin/env python3
"""
Simple threshold calculation using basic PyTorch without complex imports.
"""

import json

import numpy as np
import torch
import torch.nn.functional as F

# Test ranges
TEST_RANGES = [
    ([-2, 2], [[-6, -2], [2, 6]], "sym"),
    ([-2, -1], [-6, -2], "neg"),
    ([1, 2], [2, 6], "pos"),
    ([-1.2, -1.1], [-6.1, -1.2], "n10"),
    ([0.1, 0.2], [0.2, 2], "p01"),
    ([-0.2, -0.1], [-2, -0.2], "n01"),
    ([1.1, 1.2], [1.2, 6], "p11"),
    ([-20, -10], [-40, -20], "n20"),
    ([10, 20], [20, 40], "p20"),
]

OPERATIONS = ["add", "sub", "mul", "div"]
PERTURBATION = 1e-4


def generate_test_data(extrap_range, n_samples=10000):
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


def simple_dag_forward(input_data, operation, G_lin, G_log, O_weights):
    """
    Exact DAG forward pass matching the real implementation.
    """
    # Input: [B, 2], G_lin: [B], G_log: [B], O_weights: [B, 2]
    B = input_data.shape[0]
    clamp_min = 1e-11
    log_lim = 13.815510558  # log(1e6) - 1.0

    # Compute linear and log domain results (exactly like DAG layer)
    R_lin = torch.sum(O_weights * input_data, dim=-1, keepdim=True)  # [B, 1]

    # For log domain: work with magnitudes
    input_mag = torch.abs(input_data)
    R_log = torch.sum(
        O_weights * torch.log(torch.clamp(input_mag, min=clamp_min)),
        dim=-1,
        keepdim=True,
    )  # [B, 1]
    R_log = torch.clamp(R_log, min=-log_lim, max=log_lim)

    # Sign computation (matching actual DAG implementation)
    sign_eps = 1e-4
    linear_sign = torch.tanh(R_lin / sign_eps).squeeze(-1)  # [B] - smooth sign

    # Log domain sign computation (original DAG logic)
    w = torch.abs(O_weights)  # [B, 2]
    input_sign = torch.sign(input_data)  # [B, 2]
    neg_frac = 0.5 * (1.0 - input_sign)  # [B, 2] - fraction of negative inputs
    m = torch.sum(w * neg_frac, dim=-1)  # [B] - weighted negative fraction
    log_sign = torch.cos(torch.tensor(np.pi) * m)  # [B]

    # Single G sign mixing
    G_step = G_lin  # [B]
    V_sign_new = G_step * linear_sign + (1.0 - G_step) * log_sign  # [B]

    # Magnitude computation - CRITICAL: Mix in log space then exponentiate
    linear_mag = torch.sqrt(R_lin * R_lin + 1e-8)  # smooth |.|
    l_lin = torch.log(torch.clamp(linear_mag, min=clamp_min))  # [B, 1]
    l_log = R_log  # [B, 1]

    # Single G magnitude mixing in log space (this is the key!)
    G_step_expanded = G_step.unsqueeze(-1)  # [B, 1]
    m_log = l_log + G_step_expanded * (
        l_lin - l_log
    )  # Direct interpolation in log space
    m_log = torch.clamp(m_log, min=-log_lim, max=log_lim)
    V_mag_new = torch.exp(m_log)  # [B, 1]

    # Final result
    final_value = (V_sign_new.unsqueeze(-1) * V_mag_new).squeeze(-1)  # [B]

    return final_value.unsqueeze(-1)  # [B, 1]


def calculate_threshold_mse(operation, extrap_range, range_name):
    """Calculate threshold MSE using perfect vs perturbed G values."""

    print(f"    Generating test data...", end="", flush=True)
    test_data = generate_test_data(extrap_range, n_samples=10000)
    B = test_data.shape[0]

    # Calculate true targets
    if operation == "add":
        targets = test_data[:, 0] + test_data[:, 1]
    elif operation == "sub":
        targets = test_data[:, 0] - test_data[:, 1]
    elif operation == "mul":
        targets = test_data[:, 0] * test_data[:, 1]
    elif operation == "div":
        # Avoid division by zero
        mask = torch.abs(test_data[:, 1]) > 1e-8
        test_data = test_data[mask]
        targets = test_data[:, 0] / test_data[:, 1]

    targets = targets.unsqueeze(1)  # [B, 1]
    B = test_data.shape[0]

    # Set up O weights (perfect frozen weights)
    if operation in ["add", "mul"]:
        O_weights = torch.tensor([1.0, 1.0]).unsqueeze(0).expand(B, -1)
    else:  # sub, div
        O_weights = torch.tensor([1.0, -1.0]).unsqueeze(0).expand(B, -1)

    print(f" Running inference...", end="", flush=True)

    with torch.no_grad():
        # Perfect G values (frozen discrete values from DAG layer)
        if operation in ["add", "sub"]:
            # Linear operations: G_lin=1.0, G_log=0.0
            G_lin_perfect = torch.ones(B)
            G_log_perfect = torch.zeros(B)
        else:  # mul, div
            # Log operations: G_lin=0.0, G_log=1.0
            G_lin_perfect = torch.zeros(B)
            G_log_perfect = torch.ones(B)

        # Perturbed G values: apply perturbation AFTER getting perfect values
        # This matches the original paper methodology: W* ± ε
        if operation in ["add", "sub"]:
            # Perfect: G_lin=1.0, G_log=0.0
            # Perturb by moving G_lin away from 1.0 toward 0.5 (wrong direction)
            G_lin_perturbed = G_lin_perfect - PERTURBATION  # 1.0 - ε
            G_log_perturbed = G_log_perfect + PERTURBATION  # 0.0 + ε
        else:  # mul, div
            # Perfect: G_lin=0.0, G_log=1.0
            # Perturb by moving G_log away from 1.0 toward 0.5 (wrong direction)
            G_lin_perturbed = G_lin_perfect + PERTURBATION  # 0.0 + ε
            G_log_perturbed = G_log_perfect - PERTURBATION  # 1.0 - ε

        # Perfect predictions
        perfect_pred = simple_dag_forward(
            test_data, operation, G_lin_perfect, G_log_perfect, O_weights
        )
        perfect_mse = torch.mean((perfect_pred - targets) ** 2).item()

        # Perturbed predictions
        perturbed_pred = simple_dag_forward(
            test_data, operation, G_lin_perturbed, G_log_perturbed, O_weights
        )
        perturbed_mse = torch.mean((perturbed_pred - targets) ** 2).item()

    return {
        "perfect_mse": perfect_mse,
        "perturbed_mse": perturbed_mse,
        "threshold": perturbed_mse,
    }


def main():
    print("SIMPLE THRESHOLD CALCULATION")
    print("=" * 50)
    print(f"Using perturbation: {PERTURBATION}")
    print()

    results = {}

    for operation in OPERATIONS:
        print(f"\n🔧 Testing {operation.upper()} operation:")
        results[operation] = {}

        for interp_range, extrap_range, range_name in TEST_RANGES:
            print(f"  {range_name}: ", end="", flush=True)

            try:
                threshold_data = calculate_threshold_mse(
                    operation, extrap_range, range_name
                )
                results[operation][range_name] = threshold_data
                print(
                    f" Perfect: {threshold_data['perfect_mse']:.2e}, Threshold: {threshold_data['threshold']:.2e}"
                )
            except Exception as e:
                print(f" FAILED: {e}")
                results[operation][range_name] = None

    # Print summary
    print(f"\n📊 THRESHOLD SUMMARY:")
    print("=" * 80)
    print(f"{'Operation':<10} {'Range':<8} {'Perfect MSE':<12} {'Threshold MSE':<12}")
    print("-" * 80)

    for operation in OPERATIONS:
        for range_name in [r[2] for r in TEST_RANGES]:
            threshold_data = results[operation].get(range_name)
            if threshold_data is not None:
                perfect_mse = threshold_data["perfect_mse"]
                threshold_mse = threshold_data["threshold"]
                print(
                    f"{operation.upper():<10} {range_name:<8} {perfect_mse:<12.2e} {threshold_mse:<12.2e}"
                )
            else:
                print(
                    f"{operation.upper():<10} {range_name:<8} {'FAILED':<12} {'FAILED':<12}"
                )

    # Save results
    with open("experiment_results/simple_grokking_thresholds.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: experiment_results/simple_grokking_thresholds.json")


if __name__ == "__main__":
    main()
