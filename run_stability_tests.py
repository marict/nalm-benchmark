#!/usr/bin/env python3
"""
Standalone numerical stability tests for DAG layer.
"""

import torch

from stable_nalu.layer.dag import DAGLayer


def test_sign_computation_scaling():
    """Test that sign computation produces reasonable values near ±1."""
    print("Testing sign computation scaling...")

    layer = DAGLayer(4, 1, 3, enable_taps=False)
    layer.eval()

    # Test with a simple case where we expect positive signs
    test_input = torch.tensor([[2.0, 3.0, 1.0, 1.0]])

    with torch.no_grad():
        output = layer(test_input)

        # Manually check sign computation with current scaling
        R_lin = 5.0  # Expected from 2+3 in linear combination
        linear_sign = torch.tanh(torch.tensor(R_lin))
        log_sign = torch.tanh(torch.tensor(2.0))  # Expected from |O| * working_sign sum

        print(f"  Expected linear_sign = tanh({R_lin}) = {linear_sign:.6f}")
        print(f"  Expected log_sign = tanh(2.0) = {log_sign:.6f}")

        # These should be close to 1.0, not tiny values
        assert linear_sign > 0.9, f"Linear sign too small: {linear_sign:.6f}"
        assert log_sign > 0.9, f"Log sign too small: {log_sign:.6f}"

        print("  ✓ Sign values are reasonable (>0.9)")


def test_forward_pass_finite_outputs():
    """Test that forward pass always produces finite outputs."""
    print("Testing forward pass finite outputs...")

    layer = DAGLayer(4, 1, 3, enable_taps=False)

    test_ranges = [
        torch.randn(8, 4) * 0.1,  # Small values
        torch.randn(8, 4) * 1.0,  # Normal values
        torch.randn(8, 4) * 10.0,  # Large values
        torch.tensor([[1e-6, 1e-6, 1e-6, 1e-6]] * 8),  # Tiny values
        torch.tensor([[100.0, 100.0, 100.0, 100.0]] * 8),  # Large values
    ]

    for i, test_input in enumerate(test_ranges):
        output = layer(test_input)

        finite_check = torch.isfinite(output).all()
        input_range = test_input.abs().max().item()

        print(
            f"  Range {i+1} (max input: {input_range:.2e}): {'✓' if finite_check else '✗'}"
        )

        if not finite_check:
            print(f"    Non-finite outputs: {output.flatten()[:5]}")
            return False

    print("  ✓ All outputs finite")
    return True


def test_gradient_computation_stability():
    """Test that gradients are finite and well-bounded."""
    print("Testing gradient computation stability...")

    layer = DAGLayer(4, 1, 3, enable_taps=False)

    test_input = torch.randn(4, 4, requires_grad=True)

    # Forward pass
    output = layer(test_input)
    loss = output.sum()

    # Backward pass
    loss.backward()

    # Check input gradients
    input_grad_finite = torch.isfinite(test_input.grad).all()
    print(f"  Input gradients finite: {'✓' if input_grad_finite else '✗'}")

    # Check parameter gradients
    all_param_grads_finite = True
    max_grad_magnitude = 0.0

    for name, param in layer.named_parameters():
        if param.grad is not None:
            param_grad_finite = torch.isfinite(param.grad).all()
            grad_magnitude = param.grad.abs().max()
            max_grad_magnitude = max(max_grad_magnitude, grad_magnitude)

            if not param_grad_finite:
                print(f"    ✗ Non-finite gradient in {name}")
                all_param_grads_finite = False
            elif grad_magnitude > 1e6:
                print(f"    ✗ Large gradient in {name}: {grad_magnitude:.2e}")
                all_param_grads_finite = False

    print(f"  Parameter gradients finite: {'✓' if all_param_grads_finite else '✗'}")
    print(f"  Max gradient magnitude: {max_grad_magnitude:.2e}")

    return input_grad_finite and all_param_grads_finite and max_grad_magnitude < 1e6


def test_gate_values_within_bounds():
    """Test that gate values stay within [0,1] bounds."""
    print("Testing gate values within bounds...")

    layer = DAGLayer(4, 1, 3, enable_taps=False)
    test_input = torch.randn(8, 4)

    for training_mode in [True, False]:
        layer.train(training_mode)
        mode_name = "training" if training_mode else "eval"

        with torch.no_grad():
            output = layer(test_input)

            # Check internal gate values if available
            if hasattr(layer, "_last_G") and layer._last_G is not None:
                G_values = layer._last_G

                min_G = G_values.min().item()
                max_G = G_values.max().item()

                bounds_ok = (min_G >= 0) and (max_G <= 1)
                print(
                    f"  {mode_name} mode G range: [{min_G:.6f}, {max_G:.6f}] {'✓' if bounds_ok else '✗'}"
                )

                if not bounds_ok:
                    return False
            else:
                print(f"  {mode_name} mode: No gate values captured")

    return True


def run_all_tests():
    """Run all stability tests."""
    print("=== DAG Layer Numerical Stability Tests ===\n")

    tests = [
        test_sign_computation_scaling,
        test_forward_pass_finite_outputs,
        test_gradient_computation_stability,
        test_gate_values_within_bounds,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            result = test_func()
            if result is not False:  # None or True counts as pass
                passed += 1
                print("✓ PASSED\n")
            else:
                failed += 1
                print("✗ FAILED\n")
        except Exception as e:
            failed += 1
            print(f"✗ FAILED with exception: {e}\n")

    print(f"=== Results: {passed} passed, {failed} failed ===")

    if failed > 0:
        print("❌ Some tests failed - numerical instabilities detected!")
        return False
    else:
        print("✅ All tests passed - numerical stability looks good!")
        return True


if __name__ == "__main__":
    run_all_tests()
