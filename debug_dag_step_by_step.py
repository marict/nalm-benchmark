#!/usr/bin/env python3

import torch

from tests.test_dag_arithmetic import TestDAGArithmetic


def debug_dag_addition():
    """Debug step-by-step DAG computation for simple addition."""
    print("=== Debugging DAG Addition: 2.0 + 3.0 = 5.0 ===")

    test_instance = TestDAGArithmetic()
    test_instance.setup_method()

    # Create layer and set up for addition
    layer = test_instance.create_dag_layer(use_test_mode=True)
    layer.eval()  # Set to evaluation mode for deterministic output selection
    test_instance.manually_set_weights_for_addition(layer)

    # Use the same test inputs as the test class but examine first batch element
    test_input = test_instance.test_inputs  # This has batch size 8
    print(f"Input (batch 0): {test_input[0, :2]}  (only first 2 inputs matter)")
    print(
        f"Expected output for batch 0: {test_input[0, 0] + test_input[0, 1]} = {(test_input[0, 0] + test_input[0, 1]).item()}"
    )

    # Run forward pass
    output = layer(test_input)
    print(
        f"Final Output (batch 0): {output[0].item():.6f} (Expected: {(test_input[0, 0] + test_input[0, 1]).item():.1f})"
    )

    # Print actual computed values (not the hardcoded debug messages below)
    if hasattr(layer, "_debug_V_sign_new") and layer._debug_V_sign_new:
        actual_sign = layer._debug_V_sign_new[0][0].item()
        print(f"ACTUAL V_sign_new: {actual_sign:.6f}")
    if hasattr(layer, "_debug_V_mag_new") and layer._debug_V_mag_new:
        actual_mag = layer._debug_V_mag_new[0][0].item()
        print(f"ACTUAL V_mag_new: {actual_mag:.6f}")
        if hasattr(layer, "_debug_V_sign_new") and layer._debug_V_sign_new:
            print(f"ACTUAL Product: {actual_sign * actual_mag:.6f}")

    print("\n=== Step-by-step Analysis ===")

    # Print manual weight configuration
    print(f"Manual Weights Set:")
    print(
        f"  test_O_mag[0,0,0:2]: {layer.test_O_mag[0,0,0:2]}"
    )  # Should be [9999, 9999]
    print(
        f"  test_O_sign[0,0,0:2]: {layer.test_O_sign[0,0,0:2]}"
    )  # Should be [9999, 9999]
    print(f"  test_G[0,0]: {layer.test_G[0,0]}")  # Should be 0.0 (linear)
    print(f"  test_out_logits[0,0]: {layer.test_out_logits[0,0]}")  # Should be 9999

    # Print debug information saved during forward pass
    print(f"\nInitial working state:")
    print(f"  working_mag[0]: {layer._debug_working_mag[0][0]}")
    print(f"  working_sign[0]: {layer._debug_working_sign[0][0]}")

    print(f"\nStep 0 computation:")
    print(f"  R_lin: {layer._debug_R_lin[0][0].item():.6f}")  # Should be ~5.0 (2+3)
    print(f"  R_log: {layer._debug_R_log[0][0].item():.6f}")  # Log space result
    print(f"  V_sign_new: {layer._debug_V_sign_new[0][0].item():.6f}")  # New sign
    print(f"  V_mag_new: {layer._debug_V_mag_new[0][0].item():.6f}")  # New magnitude

    # Let's manually compute what the sign should be
    O_step = (layer.test_O_sign * layer.test_O_mag)[0, 0, :]
    working_sign = layer._debug_working_sign[0][0]
    G_step = layer.test_G[0, 0]
    R_lin = layer._debug_R_lin[0][0].item()

    linear_sign = torch.tanh(torch.tensor(R_lin * 1e-3))
    log_sign = torch.tanh((torch.abs(O_step) * working_sign).sum() * 1e-2)
    expected_sign = G_step * linear_sign + (1.0 - G_step) * log_sign

    print(f"\nSign computation details:")
    print(
        f"  linear_sign = tanh({R_lin:.3f} * 1e-3) = tanh({R_lin*1e-3:.6f}) = {linear_sign.item():.6f}"
    )
    print(f"  log_sign = tanh(sum(|O| * working_sign) * 1e-2) = {log_sign.item():.6f}")
    print(f"  G_step = {G_step:.1f} (0=linear domain)")
    print(
        f"  expected_sign = {G_step:.1f} * {linear_sign.item():.6f} + {1-G_step:.1f} * {log_sign.item():.6f} = {expected_sign.item():.6f}"
    )

    print(f"\nFinal working state (after step 0):")
    print(f"  working_mag[0]: {layer._debug_working_mag[1][0]}")
    print(f"  working_sign[0]: {layer._debug_working_sign[1][0]}")

    # Check output selection
    intermediate_values = (layer._debug_working_sign[1] * layer._debug_working_mag[1])[
        0, layer.num_initial_nodes :
    ]
    print(f"\nIntermediate values for output selection: {intermediate_values}")
    print(f"Selected value index 0: {intermediate_values[0].item():.6f}")


if __name__ == "__main__":
    debug_dag_addition()
