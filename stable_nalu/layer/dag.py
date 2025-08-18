from __future__ import annotations

import math

import torch
import torch.nn as nn

from debug_utils import tap

from ..abstract import ExtendedTorchModule


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
        use_ste_G: bool = False,
        enable_taps: bool = True,
        _do_not_predict_weights: bool = False,
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
        self.use_ste_G = bool(use_ste_G)
        self.enable_taps = bool(enable_taps)
        self._do_not_predict_weights = bool(_do_not_predict_weights)

        self.O_mag_head = nn.Linear(in_features, self.dag_depth * self.total_nodes)
        self.O_sign_head = nn.Linear(in_features, self.dag_depth * self.total_nodes)
        self.G_head = nn.Linear(in_features, self.dag_depth)

        self.O_mask = torch.zeros(self.dag_depth, self.total_nodes)
        for step in range(self.dag_depth):
            valid_nodes = self.num_initial_nodes + step
            self.O_mask[step, :valid_nodes] = 1.0

        self.output_selector_head = nn.Linear(in_features, self.dag_depth)

        self.reset_parameters()

        self._mag_min = 1e-11
        self._mag_max = 1e6
        self._log_lim = math.log(self._mag_max) - 1.0

    def reset_parameters(self) -> None:
        nn.init.normal_(self.O_mag_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.O_mag_head.bias)
        nn.init.normal_(self.O_sign_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.O_sign_head.bias)
        nn.init.normal_(self.G_head.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.G_head.bias, -2.0)
        nn.init.normal_(self.output_selector_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_selector_head.bias)

    def predict_dag_weights(self, input: torch.Tensor, device, dtype, B: int):
        """Predict DAG weights using neural network heads."""
        O_mag_flat = self.O_mag_head(input)
        O_mag_logits = O_mag_flat.view(B, self.dag_depth, self.total_nodes)
        O_mag_logits = O_mag_logits.to(dtype)
        O_mag_logits = tap(O_mag_logits, "O_mag_logits", self.enable_taps)

        O_sign_flat = self.O_sign_head(input)
        O_sign_logits = O_sign_flat.view(B, self.dag_depth, self.total_nodes)
        O_sign_logits = O_sign_logits.to(dtype)
        O_sign_logits = tap(O_sign_logits, "O_sign_logits", self.enable_taps)

        O_mask = self.O_mask.to(dtype).to(device)
        if self._is_nan("O_mag_logits", O_mag_logits) or self._is_nan(
            "O_sign_logits", O_sign_logits
        ):
            import pdb

            pdb.set_trace()

        O_t_mag = 2.0
        O_t_sign = 8.0

        O_sign_soft = torch.tanh(O_sign_logits / O_t_sign)
        O_sign_hard = (O_sign_logits >= 0).to(O_sign_logits.dtype) * 2.0 - 1.0
        O_sign = O_sign_hard + (O_sign_soft - O_sign_soft.detach())

        O_mag = torch.nn.functional.softplus(O_mag_logits / O_t_mag)
        O_mag = O_mag * O_mask
        O = O_sign * O_mag
        O = tap(O, "O_selector", self.enable_taps)

        G_t = 2.0
        eps = 1e-5
        G_logits = self.G_head(input)
        G_logits = tap(G_logits, "G_logits", self.enable_taps)
        G = torch.sigmoid(G_logits / G_t).to(dtype)
        G = eps + (1.0 - 2.0 * eps) * G
        G = tap(G, "G_gate", self.enable_taps)
        if self._is_nan("G (gate)", G):
            import pdb

            pdb.set_trace()

        if self.freeze_g_linear:
            G = torch.ones_like(G)
        if self.freeze_g_log:
            G = torch.zeros_like(G)

        if self.use_ste_G:
            G_hard = (G > 0.5).to(G.dtype)
            G = G_hard + (G - G.detach())

        if not self.training:
            G = (G > 0.5).to(G.dtype)

        # Output selector logits
        out_logits = self.output_selector_head(input).to(dtype)

        return O_mag, O_sign, G, out_logits

    @staticmethod
    def _ste_round(values: torch.Tensor) -> torch.Tensor:
        return values.round().detach() + (values - values.detach())

    def _compute_aggregates(
        self,
        working_mag: torch.Tensor,
        working_sign: torch.Tensor,
        O_step: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """L1-normalize O_step (signed convex mix), then aggregate linear/log terms."""
        signed_values = working_sign * working_mag
        log_mag = torch.log(torch.clamp(working_mag, min=self._mag_min))

        R_lin = torch.sum(O_step * signed_values, dim=-1, keepdim=True)
        R_log = torch.sum(O_step * log_mag, dim=-1, keepdim=True)

        return R_lin, R_log

    def _compute_new_sign(
        self,
        R_lin: torch.Tensor,
        working_sign: torch.Tensor,
        O_step: torch.Tensor,
        G_step: torch.Tensor,
    ) -> torch.Tensor:
        """Mix signs in linear and log domains with bounded outputs."""

        linear_sign = torch.tanh(R_lin)
        log_sign = torch.tanh(
            (torch.abs(O_step) * working_sign).sum(dim=-1, keepdim=True)
        )
        s = G_step * linear_sign + (1.0 - G_step) * log_sign
        s = torch.clamp(s, -1.0, 1.0)
        return s

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

    def _compute_new_magnitude(
        self,
        R_lin: torch.Tensor,
        R_log: torch.Tensor,
        G_step: torch.Tensor,
    ) -> torch.Tensor:
        """Mix magnitudes in log space with bounded gate leverage."""
        l_lin = torch.log(torch.clamp(torch.abs(R_lin), min=self._mag_min))
        l_log = self.soft_clamp(R_log, min=-self._log_lim, max=self._log_lim)
        delta = l_lin - l_log
        delta = tap(delta, "delta", self.enable_taps)
        # delta = torch.tanh(delta)
        # delta = tap(delta, "delta_clamped", self.enable_taps)
        m_log = l_log + G_step * delta
        m_log = torch.clamp(m_log, min=-self._log_lim, max=self._log_lim)
        V_mag = torch.exp(m_log)
        V_mag = torch.clamp(V_mag, min=self._mag_min, max=self._mag_max)
        return V_mag

    def _is_nan(self, name: str, tensor: torch.Tensor) -> None:
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

        input = tap(input, "input", self.enable_taps)
        init_sign = torch.where(
            input >= 0,
            torch.tensor(1.0, device=device),
            torch.tensor(-1.0, device=device),
        ).to(dtype)
        init_sign = tap(init_sign, "init_sign", self.enable_taps)

        if not self._do_not_predict_weights:
            # Predict weights using neural network heads
            O_mag, O_sign, G, out_logits = self.predict_dag_weights(
                input, device, dtype, B
            )
            O = O_sign * O_mag
        else:
            # Use manually set weights directly (for testing)
            O_mag = self.test_O_mag
            O_sign = self.test_O_sign
            O = O_sign * O_mag
            G = self.test_G
            out_logits = self.test_out_logits

        if self.training and not self._do_not_predict_weights:
            self._last_G = G.detach()
            self._last_O = O.detach()
            self._last_O_sign = O_sign.detach()
            self._last_O_mag = O_mag.detach()

        working_mag = torch.zeros(B, self.total_nodes, dtype=dtype, device=device)
        working_sign = torch.zeros(B, self.total_nodes, dtype=dtype, device=device)
        working_mag[:, : self.num_initial_nodes] = input
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

        for step in range(self.dag_depth):
            O_step = O[:, step, :]
            G_step = G[:, step].unsqueeze(-1)

            R_lin, R_log = self._compute_aggregates(working_mag, working_sign, O_step)
            R_lin = tap(R_lin, "R_lin", self.enable_taps)
            R_log = tap(R_log, "R_log", self.enable_taps)

            # Debug: save intermediate computation results
            self._debug_R_lin.append(R_lin.clone())
            self._debug_R_log.append(R_log.clone())

            if self._is_nan("R_lin", R_lin):
                import pdb

                pdb.set_trace()
            if self._is_nan("R_log", R_log):
                import pdb

                pdb.set_trace()
            V_sign_new = self._compute_new_sign(R_lin, working_sign, O_step, G_step)
            V_mag_new = self._compute_new_magnitude(R_lin, R_log, G_step)

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
        if not self.training:
            idx = torch.argmax(out_logits, dim=-1, keepdim=True)
            final_value = value_vec_inter.gather(-1, idx).squeeze(-1)
        else:
            probs = torch.softmax(out_logits, dim=-1)
            final_value = torch.sum(probs * value_vec_inter, dim=-1)
            self._last_out_logits = out_logits.detach()
            self._last_value_vec_inter = value_vec_inter.detach()

        final_value = tap(final_value, "final_value", self.enable_taps)
        self._is_nan("final_value", final_value)

        return final_value.to(input.dtype).unsqueeze(-1)
