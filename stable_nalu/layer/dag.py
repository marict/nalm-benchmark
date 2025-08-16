from __future__ import annotations

import torch
import torch.nn as nn

from debug_utils import tap

from ..abstract import ExtendedTorchModule

"""
grokking commands:

grokked at: 9000
python /Users/paul_curry/ai2/nalm-benchmark/experiments/single_layer_benchmark.py \
--layer-type DAG --operation div --input-size 2 --batch-size 7024 \
--max-iterations 50000 --log-interval 1000 \
--interpolation-range '[-2.0,1.0]' \
--extrapolation-range '[-6.0,-2.0]' \
--seed 1 --clip-grad-norm 0.01

"""


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
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dag_depth: int,
        writer: str = None,
        name: str | None = None,
        freeze_g_linear: bool = False,
        freeze_g_log: bool = False,
        use_ste_G: bool = True,  # Always on, otherwise NaN city.
        use_attention_selector: bool = False,
        selector_dim: int = 256,
        use_positional_encoding: bool = False,
        use_output_selector: bool = True,
        enable_taps: bool = False,  # Enable W&B logging of detailed info via taps
        **kwargs,
    ) -> None:
        super().__init__("dag", writers=writer, name=name, **kwargs)

        if out_features != 1:
            raise ValueError(
                f"DAGLayer currently supports out_features == 1, got {out_features}"
            )

        self.in_features = in_features
        self.out_features = out_features

        # Since our input is a single vector, we only need one step.
        self.dag_depth = dag_depth
        self.num_initial_nodes = in_features
        self.total_nodes = self.num_initial_nodes + self.dag_depth

        self.freeze_g_linear = bool(freeze_g_linear)
        self.freeze_g_log = bool(freeze_g_log)
        self.use_ste_G = bool(use_ste_G)
        self.enable_taps = bool(enable_taps)

        # Always-on normalizations for stability
        self.O_norm = nn.LayerNorm(self.total_nodes)

        # Small prediction head mapping input -> selector logits
        self.use_attention_selector = bool(use_attention_selector)
        self.selector_dim = int(selector_dim)
        self.use_positional_encoding = bool(use_positional_encoding)
        if self.use_attention_selector:
            # Attention-style selector: queries from input per step, keys are learned node embeddings
            self.W_q = nn.Linear(in_features, self.dag_depth * self.selector_dim)
            self.node_keys = nn.Parameter(
                torch.empty(self.total_nodes, self.selector_dim)
            )
            if self.use_positional_encoding:
                # Learned step embeddings added to each step query
                self.step_pos_embed = nn.Parameter(
                    torch.empty(self.dag_depth, self.selector_dim)
                )
            self.O_head = None
        else:
            self.O_head = nn.Linear(in_features, self.dag_depth * self.total_nodes)
        self.O_pos_head = None
        self.O_neg_head = None

        # Domain gate G in [0,1] per step: shape (dag_depth,)
        self.G_head = nn.Linear(in_features, self.dag_depth)

        # Causal mask for O: each step can only access previously computed nodes
        self.O_mask = torch.zeros(self.dag_depth, self.total_nodes)
        for step in range(self.dag_depth):
            valid_nodes = self.num_initial_nodes + step
            self.O_mask[step, :valid_nodes] = 1.0

        # Optional output selector over intermediate nodes (length == dag_depth)
        self.use_output_selector = bool(use_output_selector)
        if self.use_output_selector:
            self.output_selector_head = nn.Linear(in_features, self.dag_depth)

        # Initialize heads similar to standard small heads
        if self.O_head is not None:
            nn.init.normal_(self.O_head.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.O_head.bias)
        if self.use_attention_selector:
            nn.init.normal_(self.W_q.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.W_q.bias)
            nn.init.normal_(self.node_keys, mean=0.0, std=0.02)
            if self.use_positional_encoding:
                nn.init.normal_(self.step_pos_embed, mean=0.0, std=0.02)
        # no extra heads
        nn.init.normal_(self.G_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.G_head.bias)
        if self.use_output_selector:
            nn.init.normal_(self.output_selector_head.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.output_selector_head.bias)

        # Numerical guards
        self._mag_min = 1e-11
        self._mag_max = 1e28
        # Limit exponent to avoid overflow in float32 (exp(88) ~ 1.65e38, close to f32 max)
        # We choose 80 to provide headroom on MPS/float32 while remaining ample for float64.
        self._log_lim = 80.0

    def reset_parameters(self) -> None:
        # Reinitialize prediction heads
        if self.O_head is not None:
            nn.init.normal_(self.O_head.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.O_head.bias)
        if self.use_attention_selector:
            nn.init.normal_(self.W_q.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.W_q.bias)
            nn.init.normal_(self.node_keys, mean=0.0, std=0.02)
            if self.use_positional_encoding:
                nn.init.normal_(self.step_pos_embed, mean=0.0, std=0.02)
        if self.O_pos_head is not None:
            nn.init.normal_(self.O_pos_head.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.O_pos_head.bias)
        if self.O_neg_head is not None:
            nn.init.normal_(self.O_neg_head.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.O_neg_head.bias)
        nn.init.normal_(self.G_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.G_head.bias)
        if self.use_output_selector:
            nn.init.normal_(self.output_selector_head.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.output_selector_head.bias)
        # No output selector head; output is always the last node

    @staticmethod
    def _ste_round(values: torch.Tensor) -> torch.Tensor:
        # Straight-through rounding to nearest integer
        return values.round().detach() + (values - values.detach())

    def _compute_domain_mixed_result(
        self,
        working_mag: torch.Tensor,
        working_sign: torch.Tensor,
        O_step: torch.Tensor,
        G_step: torch.Tensor,
    ) -> torch.Tensor:
        # Mixed-domain aggregation: linear domain uses signed values; log domain uses log magnitudes
        signed_values = working_sign * working_mag
        log_mag = torch.log(torch.clamp(working_mag, min=self._mag_min))
        mixed = log_mag * (1.0 - G_step) + signed_values * G_step
        return torch.sum(O_step * mixed, dim=-1, keepdim=True)

    def _compute_new_sign(
        self,
        R_mag: torch.Tensor,
        working_sign: torch.Tensor,
        O_step: torch.Tensor,
        G_step: torch.Tensor,
    ) -> torch.Tensor:
        # Linear domain sign from result magnitude; log domain sign from product of operand signs
        linear_sign = torch.tanh(R_mag / 1e-4)
        sign_weights = (working_sign * torch.abs(O_step)) * 2.0 + 1.0
        log_sign = torch.tanh(torch.prod(sign_weights, dim=-1, keepdim=True) / 1e-4)
        return G_step * linear_sign + (1.0 - G_step) * log_sign

    def _compute_new_magnitude(
        self, R_mag: torch.Tensor, G_step: torch.Tensor
    ) -> torch.Tensor:
        # Geometric mixing in log space to tame ∂/∂G
        # l_lin = log |R|
        l_lin = torch.log(torch.clamp(torch.abs(R_mag), min=self._mag_min))
        # l_log = R clamped (already a log-magnitude)
        l_log = torch.clamp(R_mag, min=-self._log_lim, max=self._log_lim)
        # interpolate on the log scale
        m_log = G_step * l_lin + (1.0 - G_step) * l_log
        # cap then exponentiate
        m_log = torch.clamp(m_log, min=-self._log_lim, max=self._log_lim)
        return torch.exp(m_log)

    def _is_nan(self, name: str, tensor: torch.Tensor) -> None:
        # Keep legacy name but treat any non-finite as an error for clearer debugging
        if not torch.isfinite(tensor).all():
            # Find first NaN/Inf index
            finite_mask = torch.isfinite(tensor)
            bad_idx = torch.nonzero(~finite_mask, as_tuple=False)
            first_bad = bad_idx[0].tolist() if bad_idx.numel() > 0 else []

            # Get min/max of finite values only
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

            print(
                f"Non-finite detected in '{name}' (NaN/Inf). First-bad index={first_bad}; "
                f"min_finite={min_val}, max_finite={max_val}; shape={tuple(tensor.shape)}"
            )
            return True
        return False

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input: (B, in_features)
        if input.dim() != 2 or input.size(1) != self.in_features:
            raise ValueError(
                f"Expected input of shape (B, {self.in_features}), got {tuple(input.shape)}"
            )

        device = input.device
        dtype = torch.float64 if device.type != "mps" else torch.float32
        B = input.size(0)

        # Investigate this maybe we are grokking:
        # Initial node magnitudes and signs come directly from input features
        init_mag = torch.clamp(input.abs(), min=self._mag_min, max=self._mag_max).to(
            dtype
        )
        init_mag = tap(init_mag, "init_mag", self.enable_taps)
        init_sign = torch.where(
            input >= 0,
            torch.tensor(1.0, device=device),
            torch.tensor(-1.0, device=device),
        ).to(dtype)
        init_sign = tap(init_sign, "init_sign", self.enable_taps)

        # Predict structure from input
        # Soft selector with temperature; O in [-1,1]
        if self.use_attention_selector:
            # Queries per step: (B, dag_depth, selector_dim)
            q = self.W_q(input).view(B, self.dag_depth, self.selector_dim)
            if self.use_positional_encoding:
                # Broadcast add: (B, S, D) + (S, D) -> (B, S, D)
                q = q + self.step_pos_embed
            # Keys: (total_nodes, selector_dim)
            k = self.node_keys  # (N, D)
            # logits: (B, dag_depth, total_nodes)
            L = torch.einsum("bsd,nd->bsn", q, k) / (float(self.selector_dim) ** 0.5)
        else:
            O_flat = self.O_head(input)  # (B, dag_depth * total_nodes)
            L = O_flat.view(B, self.dag_depth, self.total_nodes)
        if self.O_norm is not None:
            L = self.O_norm(L)
        L = L.to(dtype)

        O_mask = self.O_mask.to(dtype).to(device)
        L = L * O_mask.unsqueeze(0)

        if self._is_nan("L (selector logits)", L):
            import pdb

            pdb.set_trace()
        sign = torch.tanh(L)
        mag = torch.sigmoid(torch.abs(L))
        O = sign * mag
        O = tap(O, "O_selector", self.enable_taps)
        if self._is_nan("O (selector)", O):
            import pdb

            pdb.set_trace()

        G_logits = self.G_head(input)  # (B, dag_depth)
        G_logits = tap(G_logits, "G_logits", self.enable_taps)
        if self._is_nan("G_logits", G_logits):
            import pdb

            pdb.set_trace()

        # Apply tanh constraint to prevent sigmoid gradient explosion
        G_logits = torch.tanh(G_logits / 10.0) * 10.0  # Smoothly constrain to [-10, 10]
        if G_logits.abs().max() > 10.1:
            import pdb

            pdb.set_trace()
        G_logits = tap(G_logits, "G_logits_tanh", self.enable_taps)

        G = torch.sigmoid(G_logits / 2.0).to(dtype)
        G = tap(G, "G_gate", self.enable_taps)
        if self._is_nan("G (gate)", G):
            import pdb

            pdb.set_trace()

        # Optionally freeze G to linear domain (G==1)
        if self.freeze_g_linear:
            G = torch.ones_like(G)
        if self.freeze_g_log:
            G = torch.zeros_like(G)

        # Optional STE discretisation in training for stability/inductive bias
        if self.use_ste_G:
            G_hard = (G > 0.5).to(G.dtype)
            G = G_hard + (G - G.detach())

        # Always harden in eval mode
        if not self.training:
            O = torch.round(O).clamp(-1.0, 1.0)
            G = (G > 0.5).to(G.dtype)

        # Expose selector tensors for external logging to avoid recomputation elsewhere
        if self.training:
            self._last_G = G.detach()
            self._last_O = O.detach()

        # Prepare working tensors: start with initial nodes, append one per step
        working_mag = torch.zeros(B, self.total_nodes, dtype=dtype, device=device)
        working_sign = torch.zeros(B, self.total_nodes, dtype=dtype, device=device)
        working_mag[:, : self.num_initial_nodes] = init_mag
        working_sign[:, : self.num_initial_nodes] = init_sign

        # Execute DAG steps
        for step in range(self.dag_depth):
            O_step = O[:, step, :]  # (B, total_nodes)
            G_step = G[:, step].unsqueeze(-1)  # (B, 1)

            R_mag = self._compute_domain_mixed_result(
                working_mag, working_sign, O_step, G_step
            )
            R_mag = tap(R_mag, "R_mag", self.enable_taps)
            self._is_nan("R_mag (mixed-domain result)", R_mag)
            V_sign_new = self._compute_new_sign(R_mag, working_sign, O_step, G_step)
            V_sign_new = tap(V_sign_new, "V_sign_new", self.enable_taps)
            V_mag_new = self._compute_new_magnitude(R_mag, G_step)
            V_mag_new = tap(V_mag_new, "V_mag_new", self.enable_taps)
            self._is_nan("V_sign_new", V_sign_new)
            self._is_nan("V_mag_new", V_mag_new)

            V_mag_new = torch.clamp(V_mag_new, min=self._mag_min, max=self._mag_max)
            V_sign_new = torch.clamp(V_sign_new, min=-1.0, max=1.0)

            idx = self.num_initial_nodes + step
            # Avoid in-place assignment that breaks autograd by using scatter to produce new tensors
            index_tensor = torch.full((B, 1), idx, device=device, dtype=torch.long)
            working_mag = working_mag.scatter(-1, index_tensor, V_mag_new)
            working_sign = working_sign.scatter(-1, index_tensor, V_sign_new)

            # Monitor working tensor health after each step
            working_mag = tap(working_mag, f"working_mag_step_{step}", self.enable_taps)
            working_sign = tap(
                working_sign, f"working_sign_step_{step}", self.enable_taps
            )

        # Final output: either last node or optional selector over intermediate nodes
        if self.use_output_selector:
            # Logits over intermediate nodes only (shape: (B, dag_depth))
            out_logits = self.output_selector_head(input).to(dtype)
            self._is_nan("out_logits (output selector)", out_logits)

            # Values for intermediate nodes slice [in_features : total_nodes) -> shape (B, dag_depth)
            value_vec_inter = (working_sign * working_mag)[:, self.num_initial_nodes :]
            # Expose for external logging
            if not self.training:
                idx = torch.argmax(out_logits, dim=-1, keepdim=True)  # (B,1)
                final_value = value_vec_inter.gather(-1, idx).squeeze(-1)
            else:
                probs = torch.softmax(out_logits, dim=-1)
                final_value = torch.sum(probs * value_vec_inter, dim=-1)
                self._last_out_logits = out_logits.detach()
                self._last_value_vec_inter = value_vec_inter.detach()
        else:
            final_idx = self.total_nodes - 1
            final_value = working_sign[:, final_idx] * working_mag[:, final_idx]

        final_value = tap(final_value, "final_value", self.enable_taps)
        self._is_nan("final_value", final_value)

        # Return with expected dtype
        return final_value.to(input.dtype).unsqueeze(-1)  # (B, 1)
