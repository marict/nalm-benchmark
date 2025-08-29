from __future__ import annotations

import math
import pdb

import torch
import torch.nn as nn

from debug_utils import tap

from ..abstract import ExtendedTorchModule

"""
python experiments/single_layer_benchmark.py \
    --layer-type DAG \
    --operation <add|mul|sub> \
    --input-size 2 \
    --batch-size 512 \
    --max-iterations 3000 \
    --learning-rate 1e-2 \
    --interpolation-range "[-2.0,2.0]" \
    --extrapolation-range "[[-6.0,-2.0],[2.0,6.0]]" \
    --no-cuda \
    --lr-cosine \
    --lr-min 1e-4 \
    --clip-grad-norm 1.0 \
    --log-interval 100

  Seed 223 with patience 100
  Results:
  - Add: Groks at 1318 steps
  - Mul: Groks at 181 steps
  - Sub: Does not grok (groks on other seeds)
  - Div: Groks at 888 steps


  Grokk with frozen add/mul
  python experiments/single_layer_benchmark.py \         nalm
    --layer-type DAG --seed 122 --no-open-browser  \
    --operation add  \
    --input-size 2 \
    --batch-size 512 \
    --max-iterations 30000 \
    --learning-rate 1e-2 \
    --interpolation-range "[1.1,1.2]" \
    --extrapolation-range "[1.2,6]" \
    --no-cuda \
    --log-interval 100 --clip-grad-norm 0.01

"""


# NOTE: Faster add convergence on no ste
# Seems to grok for add/sub/mul but not div
class DAGLayer(ExtendedTorchModule):
    """Differentiable arithmetic layer using a small learned DAG executor.

    This layer predicts a computation plan (operand selectors and domain gates)
    conditioned on the input vector and executes it with domain-mixed arithmetic
    over the input features as initial nodes.

    Design goals for NALM benchmark integration:
    - Behaves like a single arithmetic unit mapping (B, in_features) -> (B, out_features)
    - Uses only a small parametric head to predict structure; execution is analytic
    - Adds no external dependencies

    Notes:
    - For simplicity and fair comparison to single-output arithmetic units,
      this implementation currently supports out_features == 1.
    - The DAG depth defaults to in_features - 1, but can be overridden via kwarg
      'dag_depth' when constructing the layer through the GeneralizedLayer.
    - DON'T GET GRAD CLIP NORM -- helps a lot for grokking
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dag_depth: int,
        writer: str = None,
        name: str | None = None,
        _enable_debug_logging: bool = False,
        _enable_taps: bool = True,
        _do_not_predict_weights: bool = False,
        freeze_g_log: bool = False,
        freeze_g_linear: bool = False,
        freeze_O_div: bool = False,
        freeze_O_mul: bool = False,
        no_selector: bool = False,
        **kwargs,
    ) -> None:
        super().__init__("dag", writers=writer, name=name, **kwargs)

        if out_features != 1:
            raise ValueError(
                f"DAGLayer currently supports out_features == 1, got {out_features}"
            )

        self.in_features = in_features
        self.out_features = out_features

        self.dag_depth = dag_depth
        self.num_initial_nodes = in_features
        self.total_nodes = self.num_initial_nodes + self.dag_depth

        self.freeze_g_linear = bool(freeze_g_linear)
        self.freeze_g_log = bool(freeze_g_log)
        self.enable_debug_logging = bool(_enable_debug_logging)
        self.enable_taps = bool(_enable_taps)
        self._do_not_predict_weights = bool(_do_not_predict_weights)
        self.freeze_O_div = bool(freeze_O_div)
        self.freeze_O_mul = bool(freeze_O_mul)
        self.no_selector = bool(no_selector)

        head_input_size = in_features

        self.O_mag_head = nn.Linear(head_input_size, self.dag_depth * self.total_nodes)
        self.O_sign_head = nn.Linear(head_input_size, self.dag_depth * self.total_nodes)
        self.G_head = nn.Linear(head_input_size, self.dag_depth)

        # Add normalization layers (gated behind flag)
        self.input_norm = nn.LayerNorm(in_features)

        self.O_mask = torch.zeros(self.dag_depth, self.total_nodes)
        for step in range(self.dag_depth):
            valid_nodes = self.num_initial_nodes + step
            self.O_mask[step, :valid_nodes] = 1.0

        self.output_selector_head = nn.Linear(head_input_size, self.dag_depth)

        self.reset_parameters()

        self._mag_min = 1e-11
        self._mag_max = 1e6
        self._log_lim = math.log(self._mag_max) - 1.0

    def reset_parameters(self) -> None:
        # Use Xavier initialization for all weights
        nn.init.xavier_uniform_(self.O_mag_head.weight)
        nn.init.xavier_uniform_(self.O_sign_head.weight)
        nn.init.xavier_uniform_(self.G_head.weight)
        nn.init.xavier_uniform_(self.output_selector_head.weight)

        # Initialize all biases to zero
        nn.init.zeros_(self.O_mag_head.bias)
        nn.init.zeros_(self.O_sign_head.bias)
        nn.init.zeros_(self.G_head.bias)
        nn.init.zeros_(self.output_selector_head.bias)

        # Apply frozen selector initialization if enabled
        with torch.no_grad():
            if self.freeze_O_div:
                # Initialize bias for division pattern [1, -1, 0, ...]
                for step in range(self.dag_depth):
                    step_start = step * self.total_nodes
                    if self.num_initial_nodes >= 2:
                        self.O_sign_head.bias[step_start] = 1.0  # First input: positive
                        self.O_sign_head.bias[step_start + 1] = (
                            -1.0
                        )  # Second input: negative
                        # Remaining positions stay at 0

            elif self.freeze_O_mul:
                # Initialize bias for multiplication pattern [1, 1, 0, ...]
                for step in range(self.dag_depth):
                    step_start = step * self.total_nodes
                    if self.num_initial_nodes >= 2:
                        self.O_sign_head.bias[step_start] = 1.0  # First input: positive
                        self.O_sign_head.bias[step_start + 1] = (
                            1.0  # Second input: positive
                        )
                        # Remaining positions stay at 0

    def predict_dag_weights(self, input: torch.Tensor, device, dtype, B: int):
        """Predict DAG weights using neural network heads."""
        # Extract dense features if enabled
        head_input = input

        head_input = self.input_norm(head_input)  # inf, inf

        O_mag_flat = self.O_mag_head(head_input)
        O_mag_logits = O_mag_flat.view(B, self.dag_depth, self.total_nodes)
        O_mag_logits = O_mag_logits.to(dtype)
        O_mag_logits = tap(O_mag_logits, "O_mag_logits", self.enable_taps)

        O_sign_flat = self.O_sign_head(head_input)
        O_sign_logits = O_sign_flat.view(B, self.dag_depth, self.total_nodes)
        O_sign_logits = O_sign_logits.to(dtype)
        O_sign_logits = tap(O_sign_logits, "O_sign_logits", self.enable_taps)

        G_logits = self.G_head(head_input)
        G_logits = tap(G_logits, "G_logits", self.enable_taps)

        if not self.no_selector:
            out_logits = self.output_selector_head(head_input).to(dtype)
        else:
            # Create dummy out_logits for logging when no selector is used
            out_logits = torch.zeros(B, self.dag_depth, device=device, dtype=dtype)

        O_mask = self.O_mask.to(dtype).to(device)
        if (
            self._is_nan("O_mag_logits", O_mag_logits)
            or self._is_nan("O_sign_logits", O_sign_logits)
            and self.training
        ):
            pdb.set_trace()

        # Use raw logits like the working version
        O_sign = torch.tanh(O_sign_logits)
        O_mag = torch.nn.functional.softplus(O_mag_logits)

        O_mag = O_mag * O_mask
        O_sign = O_sign * O_mask
        O = O_sign * O_mag

        # Apply selector freezing if enabled
        if self.freeze_O_div:
            # Freeze to division pattern: [1, -1, 0, 0, 0, ...] for all steps
            pattern = torch.zeros(self.total_nodes, device=O.device, dtype=O.dtype)
            if self.num_initial_nodes >= 2:
                pattern[0] = 1.0  # First input: positive
                pattern[1] = -1.0  # Second input: negative
                # Remaining positions stay at 0

            # Apply pattern to all DAG steps and all batch elements
            O = pattern.unsqueeze(0).unsqueeze(0).expand(B, self.dag_depth, -1)
        elif self.freeze_O_mul:
            # Freeze to multiplication pattern: [1, 1, 0, 0, 0, ...] for all steps
            pattern = torch.zeros(self.total_nodes, device=O.device, dtype=O.dtype)
            if self.num_initial_nodes >= 2:
                pattern[0] = 1.0  # First input: positive
                pattern[1] = 1.0  # Second input: positive
                # Remaining positions stay at 0

            # Apply pattern to all DAG steps and all batch elements
            O = pattern.unsqueeze(0).unsqueeze(0).expand(B, self.dag_depth, -1)

        O = tap(O, "O_selector", self.enable_taps)

        eps = 1e-5
        G = torch.sigmoid(G_logits).to(dtype)
        G = eps + (1.0 - 2.0 * eps) * G
        G = tap(G, "G_gate", self.enable_taps)
        if self._is_nan("G (gate)", G) and self.training:
            pdb.set_trace()

        if self.freeze_g_linear:
            G = torch.ones_like(G)
        if self.freeze_g_log:
            G = torch.zeros_like(G)

        # Save raw G weights before any discretization for sparsity calculation
        raw_G = G.detach()

        if not self.training:
            # Only apply eval discretization if STEs are not already handling it
            G = (G > 0.5).to(G.dtype)
            O = torch.round(O).clamp(-1.0, 1.0)

        return O, G, out_logits, raw_G

    def _compute_simple_domain_mixed_result(
        self,
        working_mag: torch.Tensor,
        working_sign: torch.Tensor,
        O_step: torch.Tensor,
        G_step: torch.Tensor,
    ) -> torch.Tensor:
        """Simple domain mixing from the working version."""

        signed_values = working_sign * working_mag
        log_mag = torch.log(torch.clamp(working_mag, min=self._mag_min))
        mixed = log_mag * (1.0 - G_step) + signed_values * G_step
        return torch.sum(O_step * mixed, dim=-1, keepdim=True)

    def soft_floor(self, x: torch.Tensor, min: float, t: float = 1.0) -> torch.Tensor:
        beta = 1.0 / t
        return min + torch.nn.functional.softplus(x - min, beta=beta)

    def soft_ceiling(self, x: torch.Tensor, max: float, t: float = 1.0) -> torch.Tensor:
        beta = 1.0 / t
        return max - torch.nn.functional.softplus(max - x, beta=beta)

    def soft_clamp(
        self, x: torch.Tensor, min: float, max: float, t: float = 1.0
    ) -> torch.Tensor:
        return self.soft_floor(self.soft_ceiling(x, max, t), min, t)

    def _is_nan(
        self, name: str, tensor: torch.Tensor, print_debug: bool = False
    ) -> None:
        if not torch.isfinite(tensor).all():
            finite_mask = torch.isfinite(tensor)
            bad_idx = torch.nonzero(~finite_mask, as_tuple=False)
            first_bad = bad_idx[0].tolist() if bad_idx.numel() > 0 else []

            finite_vals = tensor[finite_mask]
            min_val = (
                float(finite_vals.min().item())
                if finite_vals.numel() > 0
                else float("nan")
            )
            max_val = (
                float(finite_vals.max().item())
                if finite_vals.numel() > 0
                else float("nan")
            )

            if print_debug:
                print(
                    f"Non-finite detected in '{name}' (NaN/Inf). First-bad index={first_bad}; "
                    f"min_finite={min_val}, max_finite={max_val}; shape={tuple(tensor.shape)}"
                )
            return True
        return False

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() != 2 or input.size(1) != self.in_features:
            raise ValueError(
                f"Expected input of shape (B, {self.in_features}), got {tuple(input.shape)}"
            )

        device = input.device
        dtype = torch.float64 if device.type != "mps" else torch.float32
        B = input.size(0)

        # During evaluation, greatly relax numerical limits to allow extrapolation
        if not self.training:
            orig_mag_min = self._mag_min
            orig_mag_max = self._mag_max
            orig_log_lim = self._log_lim
            self._mag_min = 1e-20  # Much smaller but not zero
            self._mag_max = 1e20  # Much larger but not infinite
            self._log_lim = 50.0  # Much larger but not infinite

        input = tap(input, "input", self.enable_taps)
        init_sign = torch.where(
            input >= 0,
            torch.tensor(1.0, device=device),
            torch.tensor(-1.0, device=device),
        ).to(dtype)
        init_sign = tap(init_sign, "init_sign", self.enable_taps)

        if not self._do_not_predict_weights:
            # Predict weights using neural network heads
            O, G, out_logits, raw_G = self.predict_dag_weights(input, device, dtype, B)
        else:
            # Use manually set weights directly (for testing)
            O_mag = self.test_O_mag
            O_sign = self.test_O_sign
            O = O_sign * O_mag
            G = self.test_G
            out_logits = self.test_out_logits
            raw_G = G.detach()  # In test mode, raw_G = G

        # Save the weights state after any hardening for logging
        # Separate tracking for training vs eval states
        if self.training:
            self._last_train_G = G.detach()
            self._last_train_O = O.detach()
            self._last_train_raw_G = raw_G.detach()
        else:
            self._last_eval_G = G.detach()
            self._last_eval_O = O.detach()
            self._last_eval_raw_G = raw_G.detach()

        working_mag = torch.zeros(B, self.total_nodes, dtype=dtype, device=device)
        working_sign = torch.zeros(B, self.total_nodes, dtype=dtype, device=device)
        working_mag[:, : self.num_initial_nodes] = abs(input)
        working_sign[:, : self.num_initial_nodes] = init_sign

        # Debug attributes to track intermediate states
        self._debug_working_mag = []
        self._debug_working_sign = []
        self._debug_R_lin = []
        self._debug_R_log = []
        self._debug_V_mag_new = []
        self._debug_V_sign_new = []

        # Save initial state
        self._debug_working_mag.append(working_mag.clone())
        self._debug_working_sign.append(working_sign.clone())
        sign_eps = 1e-4

        for step in range(self.dag_depth):
            O_step = O[:, step, :]
            G_step = G[:, step].unsqueeze(-1)

            signed_values = working_sign * working_mag
            R_lin = torch.sum(O_step * signed_values, dim=-1, keepdim=True)
            R_log = torch.sum(
                O_step * torch.log(torch.clamp(working_mag, min=self._mag_min)),
                dim=-1,
                keepdim=True,
            )
            linear_sign = torch.tanh(R_lin / sign_eps)

            # Still encode log_sign via cos
            w = torch.abs(O_step)
            neg_frac = 0.5 * (1.0 - working_sign)
            m = torch.sum(w * neg_frac, dim=-1, keepdim=True)
            log_sign = torch.cos(math.pi * m)

            V_sign_new = G_step * linear_sign + (1.0 - G_step) * log_sign

            # Magnitude computation
            # Mix magnitudes in log space for gradient stability
            linear_mag = torch.sqrt(
                R_lin * R_lin + 1e-8
            )  # smooth |.| to keep grads near 0
            l_lin = torch.log(torch.clamp(linear_mag, min=self._mag_min))
            l_log = self.soft_clamp(R_log, min=-self._log_lim, max=self._log_lim)
            m_log = l_log + G_step * (l_lin - l_log)  # convex blend in log space
            m_log = torch.clamp(m_log, min=-self._log_lim, max=self._log_lim)
            V_mag_new = torch.exp(m_log)

            # Debug logging for troubleshooting
            if (
                self.enable_debug_logging and step == 0
            ):  # Only log first step to avoid spam
                # Compute what the ideal result should be
                input_vals = (
                    working_mag[0, : self.num_initial_nodes]
                    * working_sign[0, : self.num_initial_nodes]
                )
                selector_vals = O_step[0, : self.num_initial_nodes]

                print(f"=== DEBUG STEP {step} ===")
                print(f"Input values: {input_vals.detach().cpu().numpy()}")
                print(f"Selector: {selector_vals.detach().cpu().numpy()}")
                print(
                    f"Expected linear result: {torch.sum(selector_vals * input_vals).item():.6f}"
                )
                if len(input_vals) >= 2:
                    # For division: first_input / second_input (assuming selector is [1, -1, ...])
                    target_pattern = torch.tensor(
                        [1.0, -1.0],
                        device=selector_vals.device,
                        dtype=selector_vals.dtype,
                    )
                    if torch.allclose(selector_vals[:2], target_pattern):
                        expected_div = (
                            input_vals[0] / input_vals[1]
                            if input_vals[1].abs() > 1e-6
                            else float("inf")
                        )
                        print(f"Expected division result: {expected_div.item():.6f}")

                print(
                    f"Input magnitudes: {working_mag[0, :self.num_initial_nodes].detach().cpu().numpy()}"
                )
                print(
                    f"Input signs: {working_sign[0, :self.num_initial_nodes].detach().cpu().numpy()}"
                )
                print(f"O_step: {O_step[0].detach().cpu().numpy()}")
                print(
                    f"G_step: {G_step[0].item():.6f} ({'LINEAR' if G_step[0].item() > 0.5 else 'LOG'})"
                )

                print(f"Linear sign: {linear_sign[0].item():.6f}")
                print(f"Log sign: {log_sign[0].item():.6f}")
                print(f"Final V_sign: {V_sign_new[0].item():.6f}")
                print(f"Linear mag: {linear_mag[0].item():.6f}")
                print(f"Final V_mag: {V_mag_new[0].item():.6f}")
                print(f"Final result: {(V_sign_new * V_mag_new)[0].item():.6f}")
                print("=" * 40)

            # Debug: save new computed values
            self._debug_V_sign_new.append(V_sign_new.clone())
            self._debug_V_mag_new.append(V_mag_new.clone())

            idx = self.num_initial_nodes + step
            index_tensor = torch.full((B, 1), idx, device=device, dtype=torch.long)
            working_mag = working_mag.scatter(-1, index_tensor, V_mag_new)
            working_sign = working_sign.scatter(-1, index_tensor, V_sign_new)

            # Debug: save updated working states
            self._debug_working_mag.append(working_mag.clone())
            self._debug_working_sign.append(working_sign.clone())

            working_mag = tap(working_mag, f"working_mag_step_{step}", self.enable_taps)
            working_sign = tap(
                working_sign, f"working_sign_step_{step}", self.enable_taps
            )

        self._is_nan("out_logits (output selector)", out_logits)

        value_vec_inter = (working_sign * working_mag)[:, self.num_initial_nodes :]

        if self.no_selector:
            # Skip selector and use the last intermediate node as output
            final_value = value_vec_inter[:, -1]  # Last intermediate value
        else:
            # Use the normal output selector logic
            if not self.training:
                idx = torch.argmax(out_logits, dim=-1, keepdim=True)
                final_value = value_vec_inter.gather(-1, idx).squeeze(-1)
            else:
                probs = torch.softmax(out_logits, dim=-1)
                final_value = torch.sum(probs * value_vec_inter, dim=-1)

        # Always save the current state for logging (captures clamping during eval)
        # Separate tracking for training vs eval states
        if self.training:
            self._last_train_out_logits = out_logits.detach()
            self._last_train_value_vec_inter = value_vec_inter.detach()
        else:
            self._last_eval_out_logits = out_logits.detach()
            self._last_eval_value_vec_inter = value_vec_inter.detach()

        final_value = tap(final_value, "final_value", self.enable_taps)
        self._is_nan("final_value", final_value)

        # Restore original limits after evaluation
        if not self.training:
            self._mag_min = orig_mag_min
            self._mag_max = orig_mag_max
            self._log_lim = orig_log_lim

        return final_value.to(input.dtype).unsqueeze(-1)

    def calculate_sparsity_error(self, operation: str) -> float:
        """Calculate sparsity error for dag_depth=1 only, based on G (gating) weights.

        Args:
            operation: The arithmetic operation (kept for API compatibility, not used)

        Returns:
            sparsity_error: min(|G|, |1-G|) measuring distance from G to discrete values {0,1}

        Raises:
            ValueError: If dag_depth > 1 (multiple valid solutions exist)
            RuntimeError: If model hasn't been run yet (no weights available)
        """
        if self.dag_depth > 1:
            raise ValueError(
                f"Cannot calculate sparsity error for dag_depth={self.dag_depth} > 1: "
                "multiple valid solutions exist"
            )

        # Get the most recent RAW G weights (before discretization)
        # These are the actual learned continuous values, not the post-processed discrete ones
        if hasattr(self, "_last_eval_raw_G") and self._last_eval_raw_G is not None:
            G_weights = self._last_eval_raw_G
        elif hasattr(self, "_last_train_raw_G") and self._last_train_raw_G is not None:
            G_weights = self._last_train_raw_G
        else:
            raise RuntimeError(
                "Model hasn't been run yet. Call forward() first to generate weights."
            )

        # For dag_depth=1, G has shape [batch_size, dag_depth] = [batch_size, 1]
        # Take first batch element: shape [1] -> scalar
        if G_weights.dim() == 2:
            gating_weight = G_weights[0, 0]  # Single scalar for dag_depth=1
        else:
            raise RuntimeError(f"Unexpected G weight tensor shape: {G_weights.shape}")

        # Calculate sparsity error for G (gating parameter)
        # G should be close to 0 (log domain) or 1 (linear domain) for optimal performance
        # Sparsity error = min(|G|, |1-G|) measures distance to discrete values {0, 1}
        abs_g = torch.abs(gating_weight).item()

        # Clamp to [0, 1] to handle values outside this range
        abs_g = max(0.0, min(abs_g, 1.0))

        sparsity_error = min(abs_g, 1.0 - abs_g)

        return sparsity_error
