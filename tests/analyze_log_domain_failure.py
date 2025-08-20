#!/usr/bin/env python3
"""
Analyze intermediate values in log domain to understand why operations fail.
"""

import torch

from stable_nalu.layer.dag import DAGLayer


def analyze_log_domain_operation(
    op_name, input_a, input_b, expected, manual_O_sign_a, manual_O_sign_b
):
    """Analyze a specific log domain operation step by step."""
    print(f"\n=== Analyzing {op_name}: {input_a} op {input_b} = {expected} ===")
    print(f"Expected signs: ({manual_O_sign_a:+.0f}, {manual_O_sign_b:+.0f})")

    # Create layer with manual weights for log domain
    layer = DAGLayer(4, 1, 3, enable_taps=False, _do_not_predict_weights=True)
    layer.eval()

    # Set manual weights for log domain operation
    device = next(layer.parameters()).device
    dtype = torch.float32

    layer.test_O_mag = torch.zeros(1, 3, layer.total_nodes, dtype=dtype, device=device)
    layer.test_O_sign = torch.zeros(1, 3, layer.total_nodes, dtype=dtype, device=device)
    layer.test_G = torch.zeros(1, 3, dtype=dtype, device=device)
    layer.test_out_logits = torch.zeros(1, 3, dtype=dtype, device=device)

    # Set operand selectors for step 0
    layer.test_O_mag[0, 0, 0] = 1.0  # Select input[0]
    layer.test_O_mag[0, 0, 1] = 1.0  # Select input[1]
    layer.test_O_sign[0, 0, 0] = manual_O_sign_a  # Sign for input[0]
    layer.test_O_sign[0, 0, 1] = manual_O_sign_b  # Sign for input[1]

    # Set to log domain
    layer.test_G[0, 0] = 0.0  # Log domain

    # Output selector
    layer.test_out_logits[0, 0] = 10.0  # Select first result
    layer.test_out_logits[0, 1] = -10.0
    layer.test_out_logits[0, 2] = -10.0

    # Create test input
    test_input = torch.tensor([[input_a, input_b, 0.0, 0.0]], dtype=dtype)

    # Run forward pass
    layer.train()  # To capture debug info
    with torch.no_grad():
        output = layer(test_input)
    layer.eval()

    print(f"Final output: {output.item():.6f} (expected: {expected:.1f})")

    # Analyze initial state
    print(f"\nInitial input values:")
    print(f"  input[0] = {input_a:8.3f} -> |input[0]| = {abs(input_a):8.3f}")
    print(f"  input[1] = {input_b:8.3f} -> |input[1]| = {abs(input_b):8.3f}")

    print(f"\nInitial signs (from input >= 0):")
    input_sign_0 = 1.0 if input_a >= 0 else -1.0
    input_sign_1 = 1.0 if input_b >= 0 else -1.0
    print(f"  init_sign[0] = {input_sign_0:+.0f}")
    print(f"  init_sign[1] = {input_sign_1:+.0f}")

    # Check if we have debug info
    if hasattr(layer, "_debug_working_mag") and layer._debug_working_mag:
        print(f"\n=== Step-by-step Analysis ===")

        # Initial state (before any DAG steps)
        initial_working_mag = layer._debug_working_mag[0][0]  # First batch
        initial_working_sign = layer._debug_working_sign[0][0]

        print(f"\nInitial working state:")
        print(f"  working_mag[0:4]  = {initial_working_mag[:4].numpy()}")
        print(f"  working_sign[0:4] = {initial_working_sign[:4].numpy()}")

        # Analyze each DAG step
        for step in range(min(len(layer._debug_R_lin), 1)):  # Focus on step 0
            print(f"\n--- Step {step} ---")

            # Get step values
            R_lin = layer._debug_R_lin[step][0].item()
            R_log = layer._debug_R_log[step][0].item()
            V_sign_new = layer._debug_V_sign_new[step][0].item()
            V_mag_new = layer._debug_V_mag_new[step][0].item()

            print(f"Aggregation results:")
            print(f"  R_lin = {R_lin:12.6f}")
            print(f"  R_log = {R_log:12.6f}")

            # Manual calculation of what R_lin should be
            O_combined_0 = (
                layer.test_O_sign[0, 0, 0].item() * layer.test_O_mag[0, 0, 0].item()
            )
            O_combined_1 = (
                layer.test_O_sign[0, 0, 1].item() * layer.test_O_mag[0, 0, 1].item()
            )
            expected_R_lin = O_combined_0 * input_a + O_combined_1 * input_b

            print(f"Expected R_lin calculation:")
            print(
                f"  O_combined[0] = {layer.test_O_sign[0, 0, 0].item():+.0f} * {layer.test_O_mag[0, 0, 0].item():.1f} = {O_combined_0:+.1f}"
            )
            print(
                f"  O_combined[1] = {layer.test_O_sign[0, 0, 1].item():+.0f} * {layer.test_O_mag[0, 0, 1].item():.1f} = {O_combined_1:+.1f}"
            )
            print(
                f"  Expected R_lin = {O_combined_0:+.1f} * {input_a:.1f} + {O_combined_1:+.1f} * {input_b:.1f} = {expected_R_lin:.6f}"
            )

            # Manual calculation of what R_log should be
            working_mag_0 = abs(input_a)
            working_mag_1 = abs(input_b)
            log_mag_0 = torch.log(
                torch.clamp(torch.tensor(working_mag_0), min=layer._mag_min)
            ).item()
            log_mag_1 = torch.log(
                torch.clamp(torch.tensor(working_mag_1), min=layer._mag_min)
            ).item()
            expected_R_log = (
                layer.test_O_mag[0, 0, 0].item()
                * layer.test_O_sign[0, 0, 0].item()
                * log_mag_0
                + layer.test_O_mag[0, 0, 1].item()
                * layer.test_O_sign[0, 0, 1].item()
                * log_mag_1
            )

            print(f"Expected R_log calculation:")
            print(
                f"  log(|{input_a:.1f}|) = log({working_mag_0:.1f}) = {log_mag_0:.6f}"
            )
            print(
                f"  log(|{input_b:.1f}|) = log({working_mag_1:.1f}) = {log_mag_1:.6f}"
            )
            print(
                f"  Expected R_log = {O_combined_0:+.1f} * {log_mag_0:.6f} + {O_combined_1:+.1f} * {log_mag_1:.6f} = {expected_R_log:.6f}"
            )

            # Sign computation
            print(f"\nSign computation:")
            print(f"  V_sign_new = {V_sign_new:8.6f}")

            # Since G=0 (log domain), sign comes from log_sign computation
            # log_sign = tanh((abs(O_step) * working_sign).sum())
            O_step = (layer.test_O_sign * layer.test_O_mag)[
                0, 0, :
            ]  # Step 0 operand selector
            working_sign = initial_working_sign
            log_sign_input = (torch.abs(O_step) * working_sign).sum().item()
            log_sign = torch.tanh(torch.tensor(log_sign_input)).item()

            print(f"  log_sign calculation:")
            print(f"    abs(O_step) = {torch.abs(O_step)[:4].numpy()}")
            print(f"    working_sign = {working_sign[:4].numpy()}")
            print(f"    sum(abs(O_step) * working_sign) = {log_sign_input:.6f}")
            print(f"    tanh({log_sign_input:.6f}) = {log_sign:.6f}")
            print(
                f"    With G=0: final sign = 0 * linear_sign + 1 * {log_sign:.6f} = {log_sign:.6f}"
            )

            # Magnitude computation in log domain
            print(f"\nMagnitude computation:")
            print(f"  V_mag_new = {V_mag_new:12.6f}")

            # In log domain with G=0: m_log = l_log + G * (l_lin - l_log) = l_log + 0 * delta = l_log
            l_log = torch.clamp(
                torch.tensor(R_log), min=-layer._log_lim, max=layer._log_lim
            ).item()
            expected_V_mag = torch.exp(torch.tensor(l_log)).item()
            expected_V_mag = torch.clamp(
                torch.tensor(expected_V_mag), min=layer._mag_min, max=layer._mag_max
            ).item()

            print(f"  Log domain magnitude calculation:")
            print(f"    R_log = {R_log:.6f}")
            print(
                f"    l_log (clamped) = clamp({R_log:.6f}, {-layer._log_lim:.1f}, {layer._log_lim:.1f}) = {l_log:.6f}"
            )
            print(
                f"    exp(l_log) = exp({l_log:.6f}) = {torch.exp(torch.tensor(l_log)).item():.6f}"
            )
            print(f"    V_mag_new (clamped) = {expected_V_mag:.6f}")

            # Final step result
            final_step_value = V_sign_new * V_mag_new
            print(f"\nStep {step} final value:")
            print(
                f"  sign * magnitude = {V_sign_new:.6f} * {V_mag_new:.6f} = {final_step_value:.6f}"
            )

            # After step working state
            if step + 1 < len(layer._debug_working_mag):
                final_working_mag = layer._debug_working_mag[step + 1][0]
                final_working_sign = layer._debug_working_sign[step + 1][0]

                print(f"\nWorking state after step {step}:")
                print(f"  working_mag[0:7]  = {final_working_mag[:7].numpy()}")
                print(f"  working_sign[0:7] = {final_working_sign[:7].numpy()}")

                # The new value should be at index num_initial_nodes + step
                new_idx = layer.num_initial_nodes + step
                print(
                    f"  New value at index {new_idx}: mag={final_working_mag[new_idx].item():.6f}, sign={final_working_sign[new_idx].item():.6f}"
                )

    else:
        print("No debug information available")


def main():
    """Analyze several log domain failures."""
    print("=== Log Domain Failure Analysis ===")

    # Test cases that failed in log domain
    test_cases = [
        ("pos*pos", 2.0, 3.0, 6.0, 1.0, 1.0),  # Should be log(2) + log(3) = log(6)
        ("pos/pos", 6.0, 2.0, 3.0, 1.0, -1.0),  # Should be log(6) - log(2) = log(3)
        ("neg*neg", -2.0, -3.0, 6.0, -1.0, -1.0),  # Catastrophic failure case
    ]

    for op_name, input_a, input_b, expected, sign_a, sign_b in test_cases:
        analyze_log_domain_operation(
            op_name, input_a, input_b, expected, sign_a, sign_b
        )
        print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
