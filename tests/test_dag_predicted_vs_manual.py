#!/usr/bin/env python3
"""
Unit test to compare predicted vs manual DAG weights and trace execution divergence.
"""

import torch
import torch.nn as nn

from stable_nalu.layer.dag import DAGLayer


class TestDAGPredictedVsManual:
    """Test to identify where predicted weights diverge from manual weights during execution."""

    def setup_method(self):
        """Set up test fixtures."""
        self.in_features = 4
        self.out_features = 1
        self.dag_depth = 3
        self.batch_size = 1  # Single example for easier tracing

        # Test input: -2 + 3 = 1
        self.test_input = torch.tensor([[-2.0, 3.0, 0.0, 0.0]], dtype=torch.float32)
        self.expected_output = 1.0

    def create_predicted_layer(self):
        """Create layer that uses predict_dag_weights() with forced good weights."""
        layer = DAGLayer(
            self.in_features,
            self.out_features,
            self.dag_depth,
            enable_taps=False,
            _do_not_predict_weights=False,  # Use neural network prediction
        )
        layer.eval()

        # Force the neural networks to predict good arithmetic weights
        self._force_good_predictions(layer)

        return layer

    def create_manual_layer(self):
        """Create layer that uses manual test weights."""
        layer = DAGLayer(
            self.in_features,
            self.out_features,
            self.dag_depth,
            enable_taps=False,
            _do_not_predict_weights=True,  # Use manual weights
        )
        layer.eval()

        # Set the manual weights for 2+3 addition
        self._set_manual_weights(layer)

        return layer

    def _force_good_predictions(self, layer):
        """Force neural networks to predict weights that should work for 2+3."""
        device = next(layer.parameters()).device
        dtype = torch.float32

        # We need to figure out what input logits produce the desired outputs
        # after going through the transformations in predict_dag_weights()

        with torch.no_grad():
            # For O_mag: we want [1.0, 1.0, 0, 0, ...] after softplus(logits/2.0)
            # softplus(x) ≈ x for large x, so we want logits ≈ 2.0 * desired_output
            desired_O_mag = torch.zeros(
                1, self.dag_depth, layer.total_nodes, dtype=dtype
            )
            desired_O_mag[0, 0, 0] = 1.0  # Select input[0]
            desired_O_mag[0, 0, 1] = 1.0  # Select input[1]

            # Convert desired magnitudes to logits: inverse of softplus(logits/2.0)
            # For softplus inverse: if y = softplus(x), then x = log(exp(y) - 1) ≈ y for y > 1
            O_mag_logits_target = torch.zeros_like(desired_O_mag)
            O_mag_logits_target[0, 0, 0] = (
                2.0 * 1.0
            )  # Will become ~1.0 after softplus(/2.0)
            O_mag_logits_target[0, 0, 1] = 2.0 * 1.0

            # For O_sign: we want [-1.0, +1.0, 0, 0, ...] after STE for -2 + 3
            # The hard part: (logits >= 0) * 2.0 - 1.0, so we want logits < 0 for negative, > 0 for positive
            desired_O_sign = torch.zeros(
                1, self.dag_depth, layer.total_nodes, dtype=dtype
            )
            desired_O_sign[0, 0, 0] = -1.0  # Negative sign for input[0] (-2)
            desired_O_sign[0, 0, 1] = 1.0  # Positive sign for input[1] (+3)

            O_sign_logits_target = torch.zeros_like(desired_O_sign)
            O_sign_logits_target[0, 0, 0] = -8.0  # Negative logit -> negative sign
            O_sign_logits_target[0, 0, 1] = 8.0  # Positive logit -> positive sign

            # For G: we want [1.0, ...] (linear domain) after sigmoid(logits/2.0)
            # sigmoid(x) = 1 when x >> 0, so we want large positive logits
            desired_G = torch.ones(1, self.dag_depth, dtype=dtype)  # Linear domain
            G_logits_target = (
                torch.ones(1, self.dag_depth, dtype=dtype) * 10.0
            )  # Large positive

            # For output selector: we want to select first DAG result (index 0)
            # This goes through softmax, so we want out_logits[0] >> out_logits[1,2]
            out_logits_target = torch.tensor(
                [10.0, -10.0, -10.0], dtype=dtype
            ).unsqueeze(0)

            # Now we need to solve: layer.head(input) = target_logits
            # This is tricky because we need to find weights that map our specific input to target

            # Simplified approach: directly set the output of the heads by modifying weights/bias
            input_flat = self.test_input.flatten()  # [2.0, 3.0, 0.0, 0.0]

            # For O_mag_head: want output = O_mag_logits_target.flatten()
            target_mag_flat = O_mag_logits_target.view(1, -1)  # [1, 21]
            self._set_linear_layer_output(layer.O_mag_head, input_flat, target_mag_flat)

            # For O_sign_head: want output = O_sign_logits_target.flatten()
            target_sign_flat = O_sign_logits_target.view(1, -1)  # [1, 21]
            self._set_linear_layer_output(
                layer.O_sign_head, input_flat, target_sign_flat
            )

            # For G_head: want output = G_logits_target
            self._set_linear_layer_output(layer.G_head, input_flat, G_logits_target)

            # For output_selector_head: want output = out_logits_target
            self._set_linear_layer_output(
                layer.output_selector_head, input_flat, out_logits_target
            )

    def _set_linear_layer_output(self, linear_layer, input_vec, target_output):
        """Set linear layer weights/bias to produce target output for given input."""
        with torch.no_grad():
            # For a linear layer: output = input @ weight.T + bias
            # We want: target_output = input_vec @ weight.T + bias
            # Simple solution: set bias = target_output, weight = 0
            linear_layer.bias.copy_(target_output.flatten())
            linear_layer.weight.zero_()

    def _set_manual_weights(self, layer):
        """Set manual test weights for 2+3 addition."""
        B = self.batch_size
        device = next(layer.parameters()).device
        dtype = torch.float32

        # Set manual weights directly (same as existing test)
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

        # Step 0: Select first two inputs with their natural signs for addition
        layer.test_O_mag[0, 0, 0] = 1.0  # input[0] magnitude
        layer.test_O_mag[0, 0, 1] = 1.0  # input[1] magnitude
        layer.test_O_sign[0, 0, 0] = -1.0  # input[0] negative sign (for -2)
        layer.test_O_sign[0, 0, 1] = 1.0  # input[1] positive sign (for +3)

        # Set G to linear domain (G=1.0 for addition)
        layer.test_G[0, 0] = 1.0  # Linear domain (1 means linear, 0 means log)

        # Output selector: select first intermediate node (step 0 result)
        layer.test_out_logits[0, 0] = 10.0  # Select first node
        layer.test_out_logits[0, 1] = -10.0  # Don't select second node
        layer.test_out_logits[0, 2] = -10.0  # Don't select third node

    def test_predicted_vs_manual_comparison(self):
        """Compare predicted vs manual weights and trace execution divergence."""
        print("=== DAG Predicted vs Manual Weight Comparison ===")
        print(f"Input: {self.test_input.flatten().numpy()}")
        print(
            f"Expected output: {self.expected_output} ({self.test_input[0,0].item():.1f} + {self.test_input[0,1].item():.1f})"
        )

        # Create both layers
        predicted_layer = self.create_predicted_layer()
        manual_layer = self.create_manual_layer()

        # Run forward passes and capture debug info
        print("\n--- Predicted Layer (using neural networks) ---")
        # Temporarily set to training mode to capture _last_* attributes
        predicted_layer.train()
        with torch.no_grad():
            predicted_output = predicted_layer(self.test_input)
        predicted_layer.eval()
        predicted_debug = self._extract_debug_info(predicted_layer)

        print("\n--- Manual Layer (using test weights) ---")
        manual_layer.train()
        with torch.no_grad():
            manual_output = manual_layer(self.test_input)
        manual_layer.eval()
        manual_debug = self._extract_debug_info(manual_layer)

        print(f"\nOutputs:")
        print(f"  Predicted: {predicted_output.item():.6f}")
        print(f"  Manual:    {manual_output.item():.6f}")
        print(f"  Expected:  {self.expected_output:.1f}")

        # Compare the weights that were actually used
        print(f"\n=== Weight Comparison ===")
        self._compare_weights(predicted_layer, manual_layer)

        # Also show what the neural networks are actually predicting
        print(f"\n=== Raw Neural Network Predictions ===")
        self._show_raw_predictions(predicted_layer)

        # Compare execution traces
        print(f"\n=== Execution Trace Comparison ===")
        self._compare_execution_traces(predicted_debug, manual_debug)

        # Check for divergence points
        divergence_points = self._find_divergence_points(predicted_debug, manual_debug)
        if divergence_points:
            print(f"\n⚠️  DIVERGENCE DETECTED at: {divergence_points}")
        else:
            print(f"\n✅ No significant divergence detected")

    def _extract_debug_info(self, layer):
        """Extract debug information from layer after forward pass."""
        debug_info = {}

        # Extract the weights that were actually used
        if hasattr(layer, "_last_O_mag"):
            debug_info["O_mag"] = layer._last_O_mag.clone()
        if hasattr(layer, "_last_O_sign"):
            debug_info["O_sign"] = layer._last_O_sign.clone()
        if hasattr(layer, "_last_G"):
            debug_info["G"] = layer._last_G.clone()
        if hasattr(layer, "_last_out_logits"):
            debug_info["out_logits"] = layer._last_out_logits.clone()

        # Extract execution traces
        if hasattr(layer, "_debug_working_mag"):
            debug_info["working_mag"] = [wm.clone() for wm in layer._debug_working_mag]
        if hasattr(layer, "_debug_working_sign"):
            debug_info["working_sign"] = [
                ws.clone() for ws in layer._debug_working_sign
            ]
        if hasattr(layer, "_debug_R_lin"):
            debug_info["R_lin"] = [rl.clone() for rl in layer._debug_R_lin]
        if hasattr(layer, "_debug_R_log"):
            debug_info["R_log"] = [rl.clone() for rl in layer._debug_R_log]
        if hasattr(layer, "_debug_V_sign_new"):
            debug_info["V_sign_new"] = [vs.clone() for vs in layer._debug_V_sign_new]
        if hasattr(layer, "_debug_V_mag_new"):
            debug_info["V_mag_new"] = [vm.clone() for vm in layer._debug_V_mag_new]

        return debug_info

    def _compare_weights(self, predicted_layer, manual_layer):
        """Compare the weights used by predicted vs manual layers."""

        # Get the actual weights used during execution
        pred_debug = self._extract_debug_info(predicted_layer)
        manual_debug = self._extract_debug_info(manual_layer)

        weight_names = ["O_mag", "O_sign", "G", "out_logits"]

        for name in weight_names:
            if name in pred_debug and name in manual_debug:
                pred_vals = pred_debug[name][0]  # First batch
                if manual_debug[name].dim() > 1:
                    manual_vals = manual_debug[name][0]
                else:
                    manual_vals = manual_debug[name]

                if name in ["O_mag", "O_sign"]:
                    # Only compare first step, first few nodes for operand selectors
                    pred_vals_show = pred_vals[0, :4]  # First step, first 4 nodes
                    manual_vals_show = manual_vals[0, :4]
                elif name == "G":
                    pred_vals_show = pred_vals[0]  # First step
                    manual_vals_show = manual_vals[0]
                elif name == "out_logits":
                    pred_vals_show = pred_vals
                    manual_vals_show = manual_vals

                diff = (pred_vals_show - manual_vals_show).abs().max().item()
                print(f"  {name:12s}: Predicted={pred_vals_show.detach().numpy()}")
                print(f"  {name:12s}: Manual   ={manual_vals_show.detach().numpy()}")
                print(f"  {name:12s}: Max diff ={diff:.6f}")
                print()

    def _compare_execution_traces(self, pred_debug, manual_debug):
        """Compare step-by-step execution traces."""

        trace_names = ["R_lin", "R_log", "V_sign_new", "V_mag_new"]

        for step in range(
            min(len(pred_debug.get("R_lin", [])), len(manual_debug.get("R_lin", [])))
        ):
            print(f"Step {step}:")

            for name in trace_names:
                if name in pred_debug and name in manual_debug:
                    pred_val = pred_debug[name][step][0].item()  # First batch
                    manual_val = manual_debug[name][step][0].item()
                    diff = abs(pred_val - manual_val)

                    print(
                        f"  {name:12s}: Predicted={pred_val:8.6f}, Manual={manual_val:8.6f}, Diff={diff:.6f}"
                    )
            print()

    def _find_divergence_points(self, pred_debug, manual_debug, threshold=1e-3):
        """Find where predicted and manual executions diverge significantly."""
        divergence_points = []

        # Check weights
        weight_names = ["O_mag", "O_sign", "G"]
        for name in weight_names:
            if name in pred_debug and name in manual_debug:
                pred_vals = pred_debug[name]
                manual_vals = manual_debug[name]

                if pred_vals.shape == manual_vals.shape:
                    max_diff = (pred_vals - manual_vals).abs().max().item()
                    if max_diff > threshold:
                        divergence_points.append(
                            f"{name}_weights (diff={max_diff:.6f})"
                        )

        # Check execution traces
        trace_names = ["R_lin", "R_log", "V_sign_new", "V_mag_new"]
        for step in range(
            min(len(pred_debug.get("R_lin", [])), len(manual_debug.get("R_lin", [])))
        ):
            for name in trace_names:
                if name in pred_debug and name in manual_debug:
                    pred_val = pred_debug[name][step][0].item()
                    manual_val = manual_debug[name][step][0].item()
                    diff = abs(pred_val - manual_val)

                    if diff > threshold:
                        divergence_points.append(
                            f"step_{step}_{name} (diff={diff:.6f})"
                        )

        return divergence_points

    def _show_raw_predictions(self, layer):
        """Show what the neural networks are actually predicting."""
        with torch.no_grad():
            # Get raw logits from each head
            O_mag_logits_raw = layer.O_mag_head(self.test_input)
            O_sign_logits_raw = layer.O_sign_head(self.test_input)
            G_logits_raw = layer.G_head(self.test_input)
            out_logits_raw = layer.output_selector_head(self.test_input)

            print(f"Raw head outputs for input {self.test_input.flatten().numpy()}:")
            print(
                f"  O_mag_head raw:  {O_mag_logits_raw.flatten()[:8].numpy()}"
            )  # First 8 values
            print(f"  O_sign_head raw: {O_sign_logits_raw.flatten()[:8].numpy()}")
            print(f"  G_head raw:      {G_logits_raw.flatten().numpy()}")
            print(f"  out_head raw:    {out_logits_raw.flatten().numpy()}")

            # Show after processing
            O_mag_logits = O_mag_logits_raw.view(1, layer.dag_depth, layer.total_nodes)
            O_sign_logits = O_sign_logits_raw.view(
                1, layer.dag_depth, layer.total_nodes
            )

            # Apply the transformations from predict_dag_weights
            O_mag_processed = torch.nn.functional.softplus(O_mag_logits / 2.0)
            O_sign_soft = torch.tanh(O_sign_logits / 8.0)
            O_sign_hard = (O_sign_logits >= 0).to(O_sign_logits.dtype) * 2.0 - 1.0
            O_sign_processed = O_sign_hard + (O_sign_soft - O_sign_soft.detach())
            G_processed = torch.sigmoid(G_logits_raw / 2.0)

            print(f"\nAfter processing:")
            print(f"  O_mag step0[:4]:  {O_mag_processed[0, 0, :4].numpy()}")
            print(f"  O_sign step0[:4]: {O_sign_processed[0, 0, :4].numpy()}")
            print(f"  G step0:          {G_processed[0, 0].item():.6f}")

            # Show the combined operand selector for step 0
            O_combined = O_sign_processed * O_mag_processed
            print(f"  O combined step0[:4]: {O_combined[0, 0, :4].numpy()}")


def main():
    """Run the predicted vs manual comparison test."""
    test = TestDAGPredictedVsManual()
    test.setup_method()
    test.test_predicted_vs_manual_comparison()


if __name__ == "__main__":
    main()
