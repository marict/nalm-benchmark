#!/usr/bin/env python3
"""
Comprehensive test suite for DAG layer sign computation, including:
- Arithmetic correctness for all operations
- Gradient flow through smooth parity computation
- Numerical stability and edge cases
"""

import math

import torch
import torch.nn.functional as F

from stable_nalu.layer.dag import DAGLayer


class TestDAGSignComputation:
    """Comprehensive test suite for DAG sign computation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.layer = DAGLayer(4, 1, 3, _enable_taps=False)

    def test_all_arithmetic_operations(self):
        """Test all arithmetic operations with different sign combinations."""
        print("=== Testing All Arithmetic Operations ===")

        # Test cases: (op_name, a, b, expected, O_sign_a, O_sign_b, G_domain, domain_name)
        test_cases = [
            # Addition in linear domain: ALL use (+1, +1) operand signs
            ("neg+pos", -2.0, 3.0, 1.0, 1.0, 1.0, 1.0, "Linear"),
            ("pos+neg", 3.0, -2.0, 1.0, 1.0, 1.0, 1.0, "Linear"),
            ("neg+neg", -2.0, -3.0, -5.0, 1.0, 1.0, 1.0, "Linear"),
            # Subtraction in linear domain: ALL use (+1, -1) operand signs
            ("neg-pos", -2.0, 3.0, -5.0, 1.0, -1.0, 1.0, "Linear"),
            ("pos-neg", 3.0, -2.0, 5.0, 1.0, -1.0, 1.0, "Linear"),
            ("neg-neg", -3.0, -2.0, -1.0, 1.0, -1.0, 1.0, "Linear"),
            # Multiplication in log domain: ALL use (+1, +1) operand signs
            ("neg*pos", -2.0, 3.0, -6.0, 1.0, 1.0, 0.0, "Log"),
            ("pos*neg", 2.0, -3.0, -6.0, 1.0, 1.0, 0.0, "Log"),
            ("neg*neg", -2.0, -3.0, 6.0, 1.0, 1.0, 0.0, "Log"),
            # Division in log domain: ALL use (+1, -1) operand signs
            ("neg/pos", -6.0, 3.0, -2.0, 1.0, -1.0, 0.0, "Log"),
            ("pos/neg", 6.0, -3.0, -2.0, 1.0, -1.0, 0.0, "Log"),
            ("neg/neg", -6.0, -3.0, 2.0, 1.0, -1.0, 0.0, "Log"),
        ]

        results = []
        for op_name, a, b, expected, sign_a, sign_b, G, domain_name in test_cases:
            success, error = self._test_single_operation(
                op_name, a, b, expected, sign_a, sign_b, G, domain_name
            )
            results.append((op_name, domain_name, success, error))

        # Summary
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

        linear_successes = sum(success for _, success, _ in linear_results)
        log_successes = sum(success for _, success, _ in log_results)

        print(f"\nLinear domain: {linear_successes}/{len(linear_results)} successful")
        print(f"Log domain: {log_successes}/{len(log_results)} successful")
        print(f"Total: {linear_successes + log_successes}/{len(results)} successful")

        # Check if all operations pass
        all_pass = (linear_successes == len(linear_results)) and (
            log_successes == len(log_results)
        )
        if all_pass:
            print("✅ ALL ARITHMETIC OPERATIONS PASS!")
        else:
            failed_ops = [
                name for name, success, _ in linear_results + log_results if not success
            ]
            print(f"❌ FAILED OPERATIONS: {failed_ops}")

        return all_pass

    def _test_single_operation(
        self,
        op_name,
        input_a,
        input_b,
        expected,
        manual_O_sign_a,
        manual_O_sign_b,
        manual_G,
        domain_name,
    ):
        """Test a specific operation with manual weights."""
        layer = DAGLayer(4, 1, 3, _enable_taps=False, _do_not_predict_weights=True)
        layer.eval()

        device = next(layer.parameters()).device
        dtype = torch.float32

        # Set up manual weights
        layer.test_O_mag = torch.zeros(
            1, 3, layer.total_nodes, dtype=dtype, device=device
        )
        layer.test_O_sign = torch.zeros(
            1, 3, layer.total_nodes, dtype=dtype, device=device
        )
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

        with torch.no_grad():
            output = layer(test_input)

        result = output.item()
        error = abs(result - expected)
        success = error < 0.001  # Very tight tolerance for perfect operations

        return success, error

    def test_gradient_flow(self):
        """Test that gradients flow properly through the sign computation."""
        print("\n=== Testing Gradient Flow ===")

        layer = DAGLayer(4, 1, 3, _enable_taps=False)
        layer.train()

        # Test input
        test_input = torch.tensor([[2.0, 3.0, 1.0, 1.0]], requires_grad=True)
        target = torch.tensor([[5.0]])

        # Forward pass
        output = layer(test_input)
        loss = F.mse_loss(output, target)

        # Backward pass
        loss.backward()

        # Check input gradients
        input_grad_norm = test_input.grad.norm().item()

        # Check parameter gradients
        param_grads_ok = 0
        total_params = 0
        for name, param in layer.named_parameters():
            total_params += 1
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                if grad_norm > 1e-8:
                    param_grads_ok += 1

        print(f"Input gradient norm: {input_grad_norm:.6f}")
        print(f"Parameters with good gradients: {param_grads_ok}/{total_params}")

        gradient_flow_ok = (input_grad_norm > 1e-6) and (param_grads_ok == total_params)
        if gradient_flow_ok:
            print("✅ GRADIENT FLOW PASSES")
        else:
            print("❌ GRADIENT FLOW ISSUES")

        return gradient_flow_ok

    def test_smooth_parity_computation(self):
        """Test the smooth parity computation directly."""
        print("\n=== Testing Smooth Parity Computation ===")

        # Test at a point where m ≠ 0.5 to get larger gradients
        # Let's aim for m = 0.25, so we need working_sign values that give:
        # neg_frac1 + neg_frac2 = 0.25
        # 0.5 * (1 - s1) + 0.5 * (1 - s2) = 0.25
        # => 1 - s1 - s2 = 0.5 => s1 + s2 = 1.5
        working_sign = torch.tensor([[0.9, 0.6, 0.0, 0.0]], requires_grad=True)
        O_step = torch.tensor([[1.0, 1.0, 0.0, 0.0]])

        # The smooth parity computation
        w = torch.abs(O_step)
        neg_frac = 0.5 * (1.0 - working_sign)
        m = torch.sum(w * neg_frac, dim=-1, keepdim=True)
        log_sign = torch.cos(math.pi * m)

        print(f"m: {m.item():.6f}")
        print(f"log_sign: {log_sign.item():.6f}")

        # Create loss and backprop - use target of +1 instead of 0
        loss = (log_sign - 1.0).pow(2).sum()
        loss.backward()

        grad_norm = working_sign.grad.norm().item()
        print(f"Gradient norm: {grad_norm:.6f}")

        # Should have reasonable gradients now
        smooth_parity_ok = grad_norm > 1e-6
        if smooth_parity_ok:
            print("✅ SMOOTH PARITY GRADIENTS OK")
        else:
            print("❌ SMOOTH PARITY GRADIENT ISSUES")

        return smooth_parity_ok

    def test_ste_behavior(self):
        """Test that STE preserves gradients correctly."""
        print("\n=== Testing STE Behavior ===")

        layer = DAGLayer(4, 1, 3)

        # Test values
        x = torch.tensor([-0.7, -0.2, 0.3, 0.8], requires_grad=True)

        # Apply STE
        x_ste = layer._ste_round(x)

        # Create loss and backprop
        loss = x_ste.sum()
        loss.backward()

        # Check that gradients pass through (should be all 1s for STE)
        expected_grad = torch.ones_like(x)
        grad_match = torch.allclose(x.grad, expected_grad, atol=1e-6)

        print(f"Original: {x.detach().numpy()}")
        print(f"STE: {x_ste.detach().numpy()}")
        print(f"Gradients match expected: {grad_match}")

        if grad_match:
            print("✅ STE BEHAVIOR CORRECT")
        else:
            print("❌ STE BEHAVIOR INCORRECT")

        return grad_match

    def test_training_vs_eval_modes(self):
        """Test that both training and eval modes work correctly."""
        print("\n=== Testing Training vs Eval Modes ===")

        layer = DAGLayer(4, 1, 3, _enable_taps=False)
        test_input = torch.tensor([[2.0, -3.0, 1.0, 1.0]], requires_grad=True)

        # Test training mode
        layer.train()
        output_train = layer(test_input)
        loss_train = output_train.sum()
        loss_train.backward()
        train_grad_norm = test_input.grad.norm().item()

        # Reset gradients
        test_input.grad = None

        # Test eval mode
        layer.eval()
        output_eval = layer(test_input)
        loss_eval = output_eval.sum()
        loss_eval.backward()
        eval_grad_norm = test_input.grad.norm().item()

        print(f"Training mode gradient norm: {train_grad_norm:.6f}")
        print(f"Eval mode gradient norm: {eval_grad_norm:.6f}")

        modes_ok = (train_grad_norm > 1e-6) and (eval_grad_norm > 1e-6)
        if modes_ok:
            print("✅ BOTH MODES HAVE GRADIENTS")
        else:
            print("❌ MODE ISSUES")

        return modes_ok


def run_all_tests():
    """Run all DAG sign computation tests."""
    print("DAG Layer Sign Computation - Comprehensive Test Suite")
    print("=" * 60)

    test_instance = TestDAGSignComputation()
    test_instance.setup_method()

    # Run all tests
    tests = [
        ("Arithmetic Operations", test_instance.test_all_arithmetic_operations),
        ("Gradient Flow", test_instance.test_gradient_flow),
        ("Smooth Parity", test_instance.test_smooth_parity_computation),
        ("STE Behavior", test_instance.test_ste_behavior),
        ("Training vs Eval", test_instance.test_training_vs_eval_modes),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} FAILED WITH ERROR: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("🎉 ALL TESTS PASS! DAG sign computation is working perfectly.")
    else:
        print("⚠️  Some tests failed - investigation needed.")

    return passed == len(results)


if __name__ == "__main__":
    run_all_tests()
