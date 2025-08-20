#!/usr/bin/env python3
"""
Comprehensive arithmetic test for DAG layer with different operations and sign combinations.
"""

import torch
import torch.nn as nn

from stable_nalu.layer.dag import DAGLayer


class TestDAGArithmeticComprehensive:
    """Comprehensive test for different arithmetic operations and sign combinations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.in_features = 4
        self.out_features = 1
        self.dag_depth = 3
        self.batch_size = 1

        # Test cases: (op_name, input_a, input_b, expected_result, use_natural_signs, manual_G)
        self.test_cases = [
            # Addition: use natural signs, linear domain
            ("pos+pos", 2.0, 3.0, 5.0, True, 1.0),  # Natural: (+,+) in linear domain
            ("pos+neg", 3.0, -2.0, 1.0, True, 1.0),  # Natural: (+,-) in linear domain
            ("neg+neg", -2.0, -3.0, -5.0, True, 1.0),  # Natural: (-,-) in linear domain
            # Subtraction: flip second sign, linear domain
            ("pos-pos", 5.0, 2.0, 3.0, False, 1.0),  # Manual: (+,-) for a - b
            ("pos-neg", 3.0, -2.0, 5.0, False, 1.0),  # Manual: (+,+) for a - (-b)
            ("neg-pos", -2.0, 3.0, -5.0, False, 1.0),  # Manual: (-,-) for (-a) - b
            ("neg-neg", -3.0, -2.0, -1.0, False, 1.0),  # Manual: (-,+) for (-a) - (-b)
            # Multiplication: try log domain
            ("pos*pos", 2.0, 3.0, 6.0, True, 0.0),  # Natural signs, log domain
            ("pos*neg", 2.0, -3.0, -6.0, True, 0.0),  # Natural signs, log domain
            ("neg*neg", -2.0, -3.0, 6.0, True, 0.0),  # Natural signs, log domain
            # Division: log domain with subtraction
            ("pos/pos", 6.0, 2.0, 3.0, False, 0.0),  # Manual: (+,-) for log(a) - log(b)
            (
                "pos/neg",
                6.0,
                -2.0,
                -3.0,
                False,
                0.0,
            ),  # Manual: (+,+) for log(a) - log(-b)
            (
                "neg/neg",
                -6.0,
                -2.0,
                3.0,
                False,
                0.0,
            ),  # Manual: (+,-) for log(-a) - log(-b)
        ]

    def create_predicted_layer(self, manual_O_sign_a, manual_O_sign_b, manual_G):
        """Create layer that uses predict_dag_weights() with forced weights for specific operation."""
        layer = DAGLayer(
            self.in_features,
            self.out_features,
            self.dag_depth,
            enable_taps=False,
            _do_not_predict_weights=False,
        )
        layer.eval()

        # Force the neural networks to predict the specified operation weights
        self._force_operation_predictions(
            layer, manual_O_sign_a, manual_O_sign_b, manual_G
        )

        return layer

    def create_manual_layer(self, manual_O_sign_a, manual_O_sign_b, manual_G):
        """Create layer that uses manual test weights for specific operation."""
        layer = DAGLayer(
            self.in_features,
            self.out_features,
            self.dag_depth,
            enable_taps=False,
            _do_not_predict_weights=True,
        )
        layer.eval()

        # Set the manual weights for the specified operation
        self._set_manual_operation_weights(
            layer, manual_O_sign_a, manual_O_sign_b, manual_G
        )

        return layer

    def _force_operation_predictions(
        self, layer, manual_O_sign_a, manual_O_sign_b, manual_G
    ):
        """Force neural networks to predict weights for specific arithmetic operation."""
        with torch.no_grad():
            dtype = torch.float32

            # For O_mag: we want [1.0, 1.0, 0, 0, ...]
            O_mag_logits_target = torch.zeros(
                1, self.dag_depth, layer.total_nodes, dtype=dtype
            )
            O_mag_logits_target[0, 0, 0] = 2.0  # Will become ~1.0 after softplus(/2.0)
            O_mag_logits_target[0, 0, 1] = 2.0

            # For O_sign: use the specified signs
            O_sign_logits_target = torch.zeros(
                1, self.dag_depth, layer.total_nodes, dtype=dtype
            )
            O_sign_logits_target[0, 0, 0] = 8.0 if manual_O_sign_a > 0 else -8.0
            O_sign_logits_target[0, 0, 1] = 8.0 if manual_O_sign_b > 0 else -8.0

            # For G: use specified domain (1.0 = linear, 0.0 = log)
            G_logits_target = torch.ones(1, self.dag_depth, dtype=dtype) * (
                10.0 if manual_G > 0.5 else -10.0
            )

            # For output selector: select first DAG result
            out_logits_target = torch.tensor(
                [10.0, -10.0, -10.0], dtype=dtype
            ).unsqueeze(0)

            # Set linear layer outputs directly
            input_flat = torch.zeros(4, dtype=dtype)  # Will be set per test case

            # Set up to produce target outputs regardless of input (for testing)
            layer.O_mag_head.bias.copy_(O_mag_logits_target.view(-1))
            layer.O_mag_head.weight.zero_()

            layer.O_sign_head.bias.copy_(O_sign_logits_target.view(-1))
            layer.O_sign_head.weight.zero_()

            layer.G_head.bias.copy_(G_logits_target.flatten())
            layer.G_head.weight.zero_()

            layer.output_selector_head.bias.copy_(out_logits_target.flatten())
            layer.output_selector_head.weight.zero_()

    def _set_manual_operation_weights(
        self, layer, manual_O_sign_a, manual_O_sign_b, manual_G
    ):
        """Set manual test weights for specific arithmetic operation."""
        B = self.batch_size
        device = next(layer.parameters()).device
        dtype = torch.float32

        layer.test_O_mag = torch.zeros(
            B, self.dag_depth, layer.total_nodes, dtype=dtype, device=device
        )
        layer.test_O_sign = torch.zeros(
            B, self.dag_depth, layer.total_nodes, dtype=dtype, device=device
        )
        layer.test_G = torch.zeros(B, self.dag_depth, dtype=dtype, device=device)
        layer.test_out_logits = torch.zeros(
            B, self.dag_depth, dtype=dtype, device=device
        )

        # Step 0: Select first two inputs with specified signs
        layer.test_O_mag[0, 0, 0] = 1.0  # input[0] magnitude
        layer.test_O_mag[0, 0, 1] = 1.0  # input[1] magnitude
        layer.test_O_sign[0, 0, 0] = manual_O_sign_a  # input[0] sign
        layer.test_O_sign[0, 0, 1] = manual_O_sign_b  # input[1] sign

        # Set domain (1.0 = linear, 0.0 = log)
        layer.test_G[0, 0] = manual_G

        # Output selector: select first intermediate node
        layer.test_out_logits[0, 0] = 10.0  # Select first node
        layer.test_out_logits[0, 1] = -10.0  # Don't select second node
        layer.test_out_logits[0, 2] = -10.0  # Don't select third node

    def test_all_arithmetic_operations(self):
        """Test all arithmetic operations with different sign combinations."""
        print("=== Comprehensive DAG Arithmetic Test ===\n")

        results = []

        for (
            op_name,
            input_a,
            input_b,
            expected,
            use_natural_signs,
            manual_G,
        ) in self.test_cases:
            print(f"Testing {op_name}: {input_a} op {input_b} = {expected}")

            # Create test input
            test_input = torch.tensor(
                [[input_a, input_b, 0.0, 0.0]], dtype=torch.float32
            )

            # Determine signs based on operation type
            if use_natural_signs:
                # Use natural signs from input values
                manual_O_sign_a = 1.0 if input_a >= 0 else -1.0
                manual_O_sign_b = 1.0 if input_b >= 0 else -1.0
            else:
                # Use manual signs based on operation (e.g., subtraction flips second sign)
                if "pos-pos" in op_name:
                    manual_O_sign_a, manual_O_sign_b = 1.0, -1.0
                elif "pos-neg" in op_name:
                    manual_O_sign_a, manual_O_sign_b = 1.0, 1.0  # a - (-b) = a + b
                elif "neg-pos" in op_name:
                    manual_O_sign_a, manual_O_sign_b = -1.0, -1.0  # (-a) - b = -(a + b)
                elif "neg-neg" in op_name:
                    manual_O_sign_a, manual_O_sign_b = -1.0, 1.0  # (-a) - (-b) = b - a
                elif "/pos" in op_name or "/neg" in op_name:
                    # Division: log(a) - log(b), so flip second sign
                    manual_O_sign_a = 1.0
                    manual_O_sign_b = -1.0
                else:
                    # Default fallback
                    manual_O_sign_a = 1.0 if input_a >= 0 else -1.0
                    manual_O_sign_b = 1.0 if input_b >= 0 else -1.0

            # Create both layers for this operation
            predicted_layer = self.create_predicted_layer(
                manual_O_sign_a, manual_O_sign_b, manual_G
            )
            manual_layer = self.create_manual_layer(
                manual_O_sign_a, manual_O_sign_b, manual_G
            )

            # Run predictions
            predicted_layer.train()
            with torch.no_grad():
                predicted_output = predicted_layer(test_input)
            predicted_layer.eval()

            manual_layer.train()
            with torch.no_grad():
                manual_output = manual_layer(test_input)
            manual_layer.eval()

            pred_val = predicted_output.item()
            manual_val = manual_output.item()

            pred_error = abs(pred_val - expected)
            manual_error = abs(manual_val - expected)

            domain_str = "Linear" if manual_G > 0.5 else "Log"
            sign_str = f"({manual_O_sign_a:+.0f}, {manual_O_sign_b:+.0f})"

            print(f"  Domain: {domain_str}, Signs: {sign_str}")
            print(f"  Predicted: {pred_val:7.3f} (error: {pred_error:.3f})")
            print(f"  Manual:    {manual_val:7.3f} (error: {manual_error:.3f})")
            print(f"  Expected:  {expected:7.3f}")

            # Determine success
            success_threshold = 0.5  # Allow some error
            pred_success = pred_error < success_threshold
            manual_success = manual_error < success_threshold

            status = "✅" if pred_success and manual_success else "❌"
            print(f"  Result: {status}")
            print()

            results.append(
                {
                    "op_name": op_name,
                    "domain": domain_str,
                    "pred_error": pred_error,
                    "manual_error": manual_error,
                    "pred_success": pred_success,
                    "manual_success": manual_success,
                }
            )

        # Summary
        print("=== Summary ===")
        total_tests = len(results)
        pred_successes = sum(r["pred_success"] for r in results)
        manual_successes = sum(r["manual_success"] for r in results)

        print(f"Predicted layer: {pred_successes}/{total_tests} successful")
        print(f"Manual layer: {manual_successes}/{total_tests} successful")

        # Analyze by domain
        linear_results = [r for r in results if r["domain"] == "Linear"]
        log_results = [r for r in results if r["domain"] == "Log"]

        if linear_results:
            linear_success_rate = sum(r["pred_success"] for r in linear_results) / len(
                linear_results
            )
            print(f"Linear domain success rate: {linear_success_rate:.1%}")

        if log_results:
            log_success_rate = sum(r["pred_success"] for r in log_results) / len(
                log_results
            )
            print(f"Log domain success rate: {log_success_rate:.1%}")

        # Show failures
        failures = [
            r for r in results if not (r["pred_success"] and r["manual_success"])
        ]
        if failures:
            print(f"\nFailed operations:")
            for f in failures:
                print(f"  {f['op_name']} ({f['domain']} domain)")


def main():
    """Run the comprehensive arithmetic test."""
    test = TestDAGArithmeticComprehensive()
    test.setup_method()
    test.test_all_arithmetic_operations()


if __name__ == "__main__":
    main()
