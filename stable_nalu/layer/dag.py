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
        _enable_taps: bool = False,
        _do_not_predict_weights: bool = False,
        op: str = None,
        freeze_G: bool = False,
        freeze_O: bool = False,
        no_selector: bool = False,
        unfreeze_eval: bool = False,
        G_perturbation: float = 0.0,
        freeze_input_norm: bool = False,
        use_norm: bool = True,
        single_G: bool = False,
        epsilon_smooth: bool = False,
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

        self.op = op
        self.freeze_G = bool(freeze_G)
        self.freeze_O = bool(freeze_O)
        self.enable_debug_logging = bool(_enable_debug_logging)
        self.enable_taps = bool(_enable_taps)
        self._do_not_predict_weights = bool(_do_not_predict_weights)
        self.no_selector = bool(no_selector)
        self.unfreeze_eval = bool(unfreeze_eval)
        self.G_perturbation = float(G_perturbation)
        self.freeze_input_norm = bool(freeze_input_norm)
        self.use_norm = bool(use_norm)
        self.single_G = bool(single_G)
        self.epsilon_smooth = bool(epsilon_smooth)

        head_input_size = in_features

        self.O_mag_head = nn.Linear(head_input_size, self.dag_depth * self.total_nodes)
        self.O_sign_head = nn.Linear(head_input_size, self.dag_depth * self.total_nodes)
        # G_head: outputs 2 logits per dag step if dual_G, else 1 logit per dag step
        g_head_output_size = self.dag_depth if self.single_G else self.dag_depth * 2
        self.G_head = nn.Linear(head_input_size, g_head_output_size)

        # Add normalization layers (gated behind flag)
        if self.use_norm:
            self.input_norm = nn.LayerNorm(in_features)

            # Freeze input norm parameters if specified
            if self.freeze_input_norm:
                self.input_norm.weight.requires_grad = False
                self.input_norm.bias.requires_grad = False
                # Set perfect weights: weight=1.0, bias=0.0
                with torch.no_grad():
                    self.input_norm.weight.fill_(1.0)
                    self.input_norm.bias.fill_(0.0)
        else:
            self.input_norm = None

        self.O_mask = torch.zeros(self.dag_depth, self.total_nodes)
        for step in range(self.dag_depth):
            valid_nodes = self.num_initial_nodes + step
            self.O_mask[step, :valid_nodes] = 1.0

        # Only create output selector if needed
        if not self.no_selector:
            self.output_selector_head = nn.Linear(head_input_size, self.dag_depth)
        else:
            self.output_selector_head = None

        self.reset_parameters()

        self._mag_min = 1e-11
        self._mag_max = 1e6
        self._log_lim = math.log(self._mag_max) - 1.0

    def reset_parameters(self) -> None:
        # Use Xavier initialization for all weights
        nn.init.xavier_uniform_(self.O_mag_head.weight)
        nn.init.xavier_uniform_(self.O_sign_head.weight)
        nn.init.xavier_uniform_(self.G_head.weight)

        # Initialize all biases to zero
        nn.init.zeros_(self.O_mag_head.bias)
        nn.init.zeros_(self.O_sign_head.bias)
        nn.init.zeros_(self.G_head.bias)

        # Initialize output_selector_head if it exists
        if self.output_selector_head is not None:
            nn.init.zeros_(self.output_selector_head.bias)
            nn.init.xavier_uniform_(self.output_selector_head.weight)

    def predict_dag_weights(self, input: torch.Tensor, device, dtype, B: int):
        """Predict DAG weights using neural network heads."""
        # Extract dense features if enabled
        head_input = input

        if self.use_norm:
            head_input = self.input_norm(head_input)

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
            out_logits = None

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

        # Apply selector freezing if enabled (override O with hardcoded patterns)
        if self.freeze_O:
            pattern = torch.zeros(self.total_nodes, device=O.device, dtype=O.dtype)
            if self.num_initial_nodes >= 2:
                if self.op in ["add", "mul"]:
                    # Addition and multiplication pattern: [1, 1, 0, 0, 0, ...]
                    pattern[0] = 1.0  # First input: positive
                    pattern[1] = 1.0  # Second input: positive
                elif self.op in ["sub", "div"]:
                    # Subtraction and division pattern: [1, -1, 0, 0, 0, ...]
                    pattern[0] = 1.0  # First input: positive
                    pattern[1] = -1.0  # Second input: negative
                else:
                    raise ValueError(
                        f"Unknown operation '{self.op}' for freeze_O. Must be one of: add, sub, mul, div"
                    )

            # Apply pattern to all DAG steps and all batch elements
            O = pattern.unsqueeze(0).unsqueeze(0).expand(B, self.dag_depth, -1)

        O = tap(O, "O_selector", self.enable_taps)

        if self.single_G:
            # Single G mode: G_logits has shape [B, dag_depth]
            G_single = torch.sigmoid(G_logits)  # [B, dag_depth]
            
            if self.epsilon_smooth:
                # Apply epsilon smoothing like working version for training stability
                eps = 1e-5
                G_single = eps + (1.0 - 2.0 * eps) * G_single  # Map [0,1] to [1e-5, 1-1e-5]
            
            # Store G_single directly - we'll use it in the forward pass
            G_lin = G_single  # [B, dag_depth] - this IS the gate value 
            G_log = 1.0 - G_single  # [B, dag_depth] - complement (for logging compatibility)
        else:
            # Dual G mode: G_logits has shape [B, dag_depth*2]
            # Reshape to [B, dag_depth, 2] for softmax over the 2 gate types
            G_logits_reshaped = G_logits.view(B, self.dag_depth, 2)
            G_probs = torch.softmax(G_logits_reshaped, dim=-1)  # [B, dag_depth, 2]

            # Extract linear and log gate probabilities
            G_lin = G_probs[:, :, 0]  # [B, dag_depth] - linear gate weights
            G_log = G_probs[:, :, 1]  # [B, dag_depth] - log gate weights

        # Apply frozen G values if specified (override computed values)
        if self.freeze_G:
            if self.op in ["mul", "div"]:
                # Multiplicative operations use log domain
                if self.single_G:
                    # Single G: set to prefer log domain (G_single → 0, so G_lin=0, G_log=1)
                    G_lin = torch.zeros_like(G_lin)
                    G_log = torch.ones_like(G_log)
                else:
                    # Dual G: Set G_log ≈ 1, G_lin ≈ 0
                    G_log = torch.ones_like(G_log)
                    G_lin = torch.zeros_like(G_lin)
            elif self.op in ["add", "sub"]:
                # Additive operations use linear domain (same for single_G and dual_G)
                G_lin = torch.ones_like(G_lin)
                G_log = torch.zeros_like(G_log)
            else:
                raise ValueError(
                    f"Unknown operation '{self.op}' for freeze_G. Must be one of: add, sub, mul, div"
                )

        # Apply G perturbation after freezing (if specified) - only during eval for threshold calculation
        if self.G_perturbation != 0.0 and not self.training:
            if self.op in ["add", "sub"]:
                # Linear operations: perturb G_lin down (away from 1.0) and G_log up (away from 0.0)
                G_lin = (
                    G_lin - self.G_perturbation
                )  # Push away from perfect value of 1.0
                G_log = (
                    G_log + self.G_perturbation
                )  # Push away from perfect value of 0.0
            elif self.op in ["mul", "div"]:
                # Log operations: perturb G_lin up (away from 0.0) and G_log down (away from 1.0)
                G_lin = (
                    G_lin + self.G_perturbation
                )  # Push away from perfect value of 0.0
                G_log = (
                    G_log - self.G_perturbation
                )  # Push away from perfect value of 1.0
            else:
                raise ValueError(
                    f"Unknown operation '{self.op}' for G_perturbation. Must be one of: add, sub, mul, div"
                )

            # Clamp to valid probability range [0, 1]
            G_lin = torch.clamp(G_lin, 0.0, 1.0)
            G_log = torch.clamp(G_log, 0.0, 1.0)
            
        # For backward compatibility, set G to be [G_lin, G_log] concatenated
        # This preserves the original tensor structure for logging
        G = torch.stack([G_lin, G_log], dim=-1)  # [B, dag_depth, 2]
        G = tap(G, "dual_G_gate", self.enable_taps)

        self._current_G_lin = G_lin
        self._current_G_log = G_log

        if self._is_nan("G (gate)", G) and self.training:
            pdb.set_trace()

        # Save raw G weights before any discretization for sparsity calculation
        raw_G = G.detach()

        if not self.training and not self.unfreeze_eval:
            if self.single_G:
                # For single G, discretize to 0 or 1 by thresholding at 0.5
                G_single_discrete = (self._current_G_lin > 0.5).float()
                G_lin_discrete = G_single_discrete
                G_log_discrete = 1.0 - G_single_discrete
                
                # Update current G values and reconstruct G tensor
                self._current_G_lin = G_lin_discrete
                self._current_G_log = G_log_discrete
                G = torch.stack([G_lin_discrete, G_log_discrete], dim=-1)
            else:
                # For dual_G, discretize by taking argmax and converting to one-hot
                max_indices = torch.argmax(G, dim=-1)  # [B, dag_depth]
                G_discrete = torch.zeros_like(G)  # [B, dag_depth, 2]
                G_discrete.scatter_(-1, max_indices.unsqueeze(-1), 1.0)
                G = G_discrete

                # Update the current G values for computation
                self._current_G_lin = G[:, :, 0]
                self._current_G_log = G[:, :, 1]

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

            # Use separate linear and log gate weights
            G_lin_step = self._current_G_lin[:, step].unsqueeze(-1)  # [B, 1]
            G_log_step = self._current_G_log[:, step].unsqueeze(-1)  # [B, 1]

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

            # Mix signs based on gate mode
            if self.single_G:
                # Single G: convex blend (matching working version)
                G_step = G_lin_step  # G_single stored in G_lin
                V_sign_new = G_step * linear_sign + (1.0 - G_step) * log_sign
            else:
                # Dual G: weighted combination
                V_sign_new = G_lin_step * linear_sign + G_log_step * log_sign

            # Magnitude computation
            # Mix magnitudes in log space for gradient stability
            linear_mag = torch.sqrt(
                R_lin * R_lin + 1e-8
            )  # smooth |.| to keep grads near 0
            l_lin = torch.log(torch.clamp(linear_mag, min=self._mag_min))
            l_log = self.soft_clamp(R_log, min=-self._log_lim, max=self._log_lim)

            # Use convex blend in log space (matching working version)
            if self.single_G:
                # Single G: convex blend in log space with G_single as the blend factor
                G_step = G_lin_step  # G_single stored in G_lin
                m_log = l_log + G_step * (l_lin - l_log)  # Direct interpolation in log space
            else:
                # Dual G: use weighted combination in log space
                m_log = G_log_step * l_log + G_lin_step * l_lin
            
            m_log = torch.clamp(m_log, min=-self._log_lim, max=self._log_lim)
            V_mag_new = torch.exp(m_log)

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

        if out_logits is not None:
            self._is_nan("out_logits (output selector)", out_logits)

        value_vec_inter = (working_sign * working_mag)[:, self.num_initial_nodes :]

        if self.no_selector:
            # Skip selector and use the last intermediate node as output
            final_value = value_vec_inter[:, -1]  # Last intermediate value
        else:
            # Use the normal output selector logic
            if not self.training and not self.unfreeze_eval:
                idx = torch.argmax(out_logits, dim=-1, keepdim=True)
                final_value = value_vec_inter.gather(-1, idx).squeeze(-1)
            else:
                probs = torch.softmax(out_logits, dim=-1)
                final_value = torch.sum(probs * value_vec_inter, dim=-1)

        # Always save the current state for logging (captures clamping during eval)
        # Separate tracking for training vs eval states
        if self.training:
            self._last_train_out_logits = (
                out_logits.detach() if out_logits is not None else None
            )
            self._last_train_value_vec_inter = value_vec_inter.detach()
        else:
            self._last_eval_out_logits = (
                out_logits.detach() if out_logits is not None else None
            )
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
