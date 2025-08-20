#!/usr/bin/env python3
"""
Test all negative number operations to see which ones are broken.
"""

import torch

from stable_nalu.layer.dag import DAGLayer


def test_negative_operation(
    op_name,
    input_a,
    input_b,
    expected,
    manual_O_sign_a,
    manual_O_sign_b,
    manual_G,
    domain_name,
):
    """Test a specific operation with negative numbers."""
    print(
        f"\n=== {op_name}: {input_a} op {input_b} = {expected} ({domain_name} domain) ==="
    )

    # Create layer with manual weights
    layer = DAGLayer(4, 1, 3, enable_taps=False, _do_not_predict_weights=True)
    layer.eval()

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

    # Set domain
    layer.test_G[0, 0] = manual_G

    # Output selector
    layer.test_out_logits[0, 0] = 10.0
    layer.test_out_logits[0, 1] = -10.0
    layer.test_out_logits[0, 2] = -10.0

    # Create test input and run
    test_input = torch.tensor([[input_a, input_b, 0.0, 0.0]], dtype=dtype)

    layer.train()  # To capture debug info
    with torch.no_grad():
        output = layer(test_input)
    layer.eval()

    result = output.item()
    error = abs(result - expected)
    error_pct = (error / abs(expected)) * 100 if expected != 0 else error * 100

    success = error < 0.5  # Allow some tolerance
    status = "✅" if success else "❌"

    print(f"  Manual signs: ({manual_O_sign_a:+.0f}, {manual_O_sign_b:+.0f})")
    print(f"  Result: {result:8.3f}")
    print(f"  Expected: {expected:8.3f}")
    print(f"  Error: {error:.3f} ({error_pct:.1f}%) {status}")

    # Show key intermediate values if available
    if hasattr(layer, "_debug_R_lin") and layer._debug_R_lin:
        R_lin = layer._debug_R_lin[0][0].item()
        R_log = layer._debug_R_log[0][0].item()
        V_sign_new = layer._debug_V_sign_new[0][0].item()
        V_mag_new = layer._debug_V_mag_new[0][0].item()

        print(f"  R_lin: {R_lin:8.3f}, R_log: {R_log:8.3f}")
        print(f"  V_sign: {V_sign_new:7.3f}, V_mag: {V_mag_new:8.3f}")
        print(f"  Product: {V_sign_new * V_mag_new:8.3f}")

    return success, error


def main():
    """Test all negative number operations systematically."""
    print("=== Testing Negative Number Operations ===")

    # Test cases: (op_name, a, b, expected, O_sign_a, O_sign_b, G_domain, domain_name)
    # CORRECTED: O_sign should encode OPERATION TYPE, not input signs
    test_cases = [
        # Addition in linear domain: ALL use (+1, +1) operand signs
        ("neg+pos", -2.0, 3.0, 1.0, 1.0, 1.0, 1.0, "Linear"),  # Addition always (+1,+1)
        ("pos+neg", 3.0, -2.0, 1.0, 1.0, 1.0, 1.0, "Linear"),  # Addition always (+1,+1)
        (
            "neg+neg",
            -2.0,
            -3.0,
            -5.0,
            1.0,
            1.0,
            1.0,
            "Linear",
        ),  # Addition always (+1,+1)
        # Subtraction in linear domain: ALL use (+1, -1) operand signs
        (
            "neg-pos",
            -2.0,
            3.0,
            -5.0,
            1.0,
            -1.0,
            1.0,
            "Linear",
        ),  # Subtraction always (+1,-1)
        (
            "pos-neg",
            3.0,
            -2.0,
            5.0,
            1.0,
            -1.0,
            1.0,
            "Linear",
        ),  # Subtraction always (+1,-1)
        (
            "neg-neg",
            -3.0,
            -2.0,
            -1.0,
            1.0,
            -1.0,
            1.0,
            "Linear",
        ),  # Subtraction always (+1,-1)
        # Multiplication in log domain: ALL use (+1, +1) operand signs
        (
            "neg*pos",
            -2.0,
            3.0,
            -6.0,
            1.0,
            1.0,
            0.0,
            "Log",
        ),  # Multiplication always (+1,+1)
        (
            "pos*neg",
            2.0,
            -3.0,
            -6.0,
            1.0,
            1.0,
            0.0,
            "Log",
        ),  # Multiplication always (+1,+1)
        (
            "neg*neg",
            -2.0,
            -3.0,
            6.0,
            1.0,
            1.0,
            0.0,
            "Log",
        ),  # Multiplication always (+1,+1)
        # Division in log domain: ALL use (+1, -1) operand signs
        ("neg/pos", -6.0, 3.0, -2.0, 1.0, -1.0, 0.0, "Log"),  # Division always (+1,-1)
        ("pos/neg", 6.0, -3.0, -2.0, 1.0, -1.0, 0.0, "Log"),  # Division always (+1,-1)
        ("neg/neg", -6.0, -3.0, 2.0, 1.0, -1.0, 0.0, "Log"),  # Division always (+1,-1)
    ]

    results = []

    for op_name, a, b, expected, sign_a, sign_b, G, domain_name in test_cases:
        success, error = test_negative_operation(
            op_name, a, b, expected, sign_a, sign_b, G, domain_name
        )
        results.append((op_name, domain_name, success, error))

    # Summary
    print(f"\n=== Summary ===")
    linear_results = [
        (name, success, error)
        for name, domain, success, error in results
        if domain == "Linear"
    ]
    log_results = [
        (name, success, error)
        for name, domain, success, error in results
        if domain == "Log"
    ]

    print(f"\nLinear Domain Results:")
    linear_successes = 0
    for name, success, error in linear_results:
        status = "✅" if success else "❌"
        print(f"  {name:12s}: {status} (error: {error:.3f})")
        if success:
            linear_successes += 1
    print(f"Linear domain: {linear_successes}/{len(linear_results)} successful")

    print(f"\nLog Domain Results:")
    log_successes = 0
    for name, success, error in log_results:
        status = "✅" if success else "❌"
        print(f"  {name:15s}: {status} (error: {error:.3f})")
        if success:
            log_successes += 1
    print(f"Log domain: {log_successes}/{len(log_results)} successful")

    # Identify patterns
    print(f"\n=== Analysis ===")
    broken_ops = [name for name, domain, success, error in results if not success]
    if broken_ops:
        print(f"Broken operations: {', '.join(broken_ops)}")
    else:
        print("All operations working!")


if __name__ == "__main__":
    main()
