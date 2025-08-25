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


class AttentionDAGPredictor(nn.Module):
    """Heavily parameterized attention-based predictor for DAG structure.

    Uses multi-head attention to analyze input relationships and predict
    operand selectors, domain gates, and output selectors based on learned
    patterns in the input vector.
    """

    def __init__(
        self,
        input_size: int,  # size per feature (1 for raw input, 8 for dense features)
        num_features: int,  # number of input features
        dag_depth: int,
        total_nodes: int,
        d_model: int = 256,  # attention hidden dimension
        num_heads: int = 8,  # multi-head attention heads
        num_layers: int = 4,  # transformer layers
    ):
        super().__init__()

        self.input_size = input_size
        self.num_features = num_features
        self.dag_depth = dag_depth
        self.total_nodes = total_nodes
        self.d_model = d_model

        # Project input features to attention dimension
        self.input_projection = nn.Linear(input_size, d_model)

        # Multi-head attention layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projections for DAG components
        self.O_mag_projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, dag_depth * total_nodes),
        )

        self.O_sign_projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, dag_depth * total_nodes),
        )

        self.G_projector = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, dag_depth)
        )

        self.output_selector_projector = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, dag_depth)
        )

        # Global pooling for sequence-to-vector
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(
        self, head_input: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            head_input: (B, num_features * input_size) - flattened input features

        Returns:
            O_mag_logits: (B, dag_depth * total_nodes)
            O_sign_logits: (B, dag_depth * total_nodes)
            G_logits: (B, dag_depth)
            out_logits: (B, dag_depth)
        """
        B = head_input.size(0)

        # Reshape to per-feature representation: (B, num_features, input_size)
        feature_input = head_input.view(B, self.num_features, self.input_size)

        # Project to attention dimension: (B, num_features, d_model)
        projected = self.input_projection(feature_input)

        # Apply transformer attention: (B, num_features, d_model)
        attended = self.transformer(projected)

        # Global pooling to get single representation: (B, d_model)
        # attended: (B, num_features, d_model) -> (B, d_model, num_features)
        pooled = self.global_pool(attended.transpose(-1, -2)).squeeze(-1)

        # Generate DAG predictions
        O_mag_logits = self.O_mag_projector(pooled)
        O_sign_logits = self.O_sign_projector(pooled)
        G_logits = self.G_projector(pooled)
        out_logits = self.output_selector_projector(pooled)

        return O_mag_logits, O_sign_logits, G_logits, out_logits


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
        use_simple_domain_mixing: bool = True,
        use_attention_prediction: bool = False,
        use_dense_features: bool = False,
        extended_mul_features: bool = False,
        _enable_debug_logging: bool = False,
        _enable_taps: bool = True,
        _do_not_predict_weights: bool = False,
        freeze_g_log: bool = False,
        freeze_g_linear: bool = False,
        freeze_O_div: bool = False,
        freeze_O_mul: bool = False,
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
        self.use_simple_domain_mixing = bool(use_simple_domain_mixing)
        self.use_attention_prediction = bool(use_attention_prediction)
        self.enable_debug_logging = bool(_enable_debug_logging)
        self.enable_taps = bool(_enable_taps)
        self._do_not_predict_weights = bool(_do_not_predict_weights)
        self.use_dense_features = bool(use_dense_features)
        self.extended_mul_features = bool(extended_mul_features)
        self.freeze_O_div = bool(freeze_O_div)
        self.freeze_O_mul = bool(freeze_O_mul)

        # Calculate input size for heads based on dense features
        if self.use_dense_features:
            if self.extended_mul_features:
                # Extended features: base 8 + extended 9 = 17 per input
                # Extended: [x^4, x^5, x^(-1), x^(1/2), x^(1/3), log2(|x|), log10(|x|), 2^x, 10^x]
                self.dense_features_per_input = 17
            else:
                # Dense features: [x, x^2, x^3, exp(x), log(|x|+eps), sin(x), cos(x), tanh(x)] per input
                self.dense_features_per_input = 8
            head_input_size = in_features * self.dense_features_per_input
        else:
            head_input_size = in_features

        self.O_mag_head = nn.Linear(head_input_size, self.dag_depth * self.total_nodes)
        self.O_sign_head = nn.Linear(head_input_size, self.dag_depth * self.total_nodes)
        self.G_head = nn.Linear(head_input_size, self.dag_depth)

        # Add normalization layers (gated behind flag)
        self.input_norm = nn.LayerNorm(in_features)
        self.head_norm = nn.LayerNorm(head_input_size)
        self.extra_norm1 = nn.LayerNorm(head_input_size)
        self.extra_norm2 = nn.LayerNorm(head_input_size)
        self.extra_norm3 = nn.LayerNorm(head_input_size)
        self.extra_norm4 = nn.LayerNorm(head_input_size)
        self.extra_norm5 = nn.LayerNorm(head_input_size)
        self.extra_norm6 = nn.LayerNorm(head_input_size)
        self.extra_norm7 = nn.LayerNorm(head_input_size)
        self.extra_norm8 = nn.LayerNorm(head_input_size)
        self.extra_norm9 = nn.LayerNorm(head_input_size)
        self.extra_norm10 = nn.LayerNorm(head_input_size)

        self.O_norm = nn.LayerNorm(self.total_nodes)

        self.O_mask = torch.zeros(self.dag_depth, self.total_nodes)
        for step in range(self.dag_depth):
            valid_nodes = self.num_initial_nodes + step
            self.O_mask[step, :valid_nodes] = 1.0

        self.output_selector_head = nn.Linear(head_input_size, self.dag_depth)

        # Attention-based DAG predictor (optional)
        if self.use_attention_prediction:
            self.attention_predictor = AttentionDAGPredictor(
                input_size=head_input_size // in_features,  # per-feature size
                num_features=in_features,
                dag_depth=dag_depth,
                total_nodes=self.total_nodes,
                d_model=256,  # attention hidden size
                num_heads=8,  # multi-head attention
                num_layers=4,  # number of transformer layers
            )

        self.reset_parameters()

        self._mag_min = 1e-11
        self._mag_max = 1e11
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
                        self.O_sign_head.bias[step_start] = 1.0      # First input: positive
                        self.O_sign_head.bias[step_start + 1] = -1.0  # Second input: negative
                        # Remaining positions stay at 0
            
            elif self.freeze_O_mul:
                # Initialize bias for multiplication pattern [1, 1, 0, ...]
                for step in range(self.dag_depth):
                    step_start = step * self.total_nodes
                    if self.num_initial_nodes >= 2:
                        self.O_sign_head.bias[step_start] = 1.0      # First input: positive
                        self.O_sign_head.bias[step_start + 1] = 1.0  # Second input: positive
                        # Remaining positions stay at 0


    def extract_dense_features(self, input: torch.Tensor) -> torch.Tensor:
        """Extract dense features from each input element.

        Standard: [x, x^2, x^3, exp(x), log(|x|+eps), sin(x), cos(x), tanh(x)]
        Extended: adds [x^4, x^5, x^(-1), x^(1/2), x^(1/3), log2(|x|+eps), log10(|x|+eps), 2^x, 10^x]
        """
        eps = 1e-6

        # Basic polynomial features
        x = input
        x2 = x * x
        x3 = x2 * x

        # Exponential and logarithmic features (with clamping for stability)
        x_clamped = torch.clamp(x, min=-10.0, max=10.0)  # Prevent exp overflow
        exp_x = torch.exp(x_clamped)
        log_abs_x = torch.log(torch.abs(x) + eps)

        # Trigonometric features
        sin_x = torch.sin(x)
        cos_x = torch.cos(x)

        # Hyperbolic feature
        tanh_x = torch.tanh(x)

        if self.extended_mul_features:
            # Extended polynomial features
            x4 = x3 * x
            x5 = x4 * x
            x_inv = 1.0 / (torch.abs(x) + eps)  # x^(-1), prevent division by zero
            x_sqrt = torch.sign(x) * torch.sqrt(
                torch.abs(x) + eps
            )  # x^(1/2), preserve sign
            x_cbrt = torch.sign(x) * torch.pow(
                torch.abs(x) + eps, 1.0 / 3.0
            )  # x^(1/3), preserve sign

            # Extended logarithmic features
            log2_abs_x = torch.log2(torch.abs(x) + eps)
            log10_abs_x = torch.log10(torch.abs(x) + eps)

            # Extended exponential features (with additional clamping)
            x_very_clamped = torch.clamp(
                x, min=-5.0, max=5.0
            )  # More conservative for large bases
            exp2_x = torch.pow(2.0, x_very_clamped)
            exp10_x = torch.pow(10.0, x_very_clamped)

            # Stack all features (17 total)
            dense_features = torch.stack(
                [
                    x,
                    x2,
                    x3,
                    exp_x,
                    log_abs_x,
                    sin_x,
                    cos_x,
                    tanh_x,  # Original 8
                    x4,
                    x5,
                    x_inv,
                    x_sqrt,
                    x_cbrt,
                    log2_abs_x,
                    log10_abs_x,
                    exp2_x,
                    exp10_x,  # Extended 9
                ],
                dim=-1,
            )
        else:
            # Original features (8 total)
            dense_features = torch.stack(
                [x, x2, x3, exp_x, log_abs_x, sin_x, cos_x, tanh_x], dim=-1
            )

        # Flatten to (B, in_features * dense_features_per_input)
        B, in_features, _ = dense_features.shape
        return dense_features.view(B, in_features * self.dense_features_per_input)

    def predict_dag_weights(self, input: torch.Tensor, device, dtype, B: int):
        """Predict DAG weights using neural network heads."""
        # Extract dense features if enabled
        if self.use_dense_features:
            head_input = self.extract_dense_features(input)
        else:
            head_input = input

        if self.use_attention_prediction:
            # Use attention-based predictor
            O_mag_flat, O_sign_flat, G_logits, out_logits = self.attention_predictor(
                head_input
            )

            O_mag_logits = O_mag_flat.view(B, self.dag_depth, self.total_nodes)
            O_mag_logits = O_mag_logits.to(dtype)
            O_mag_logits = tap(O_mag_logits, "O_mag_logits", self.enable_taps)

            O_sign_logits = O_sign_flat.view(B, self.dag_depth, self.total_nodes)
            O_sign_logits = O_sign_logits.to(dtype)
            O_sign_logits = tap(O_sign_logits, "O_sign_logits", self.enable_taps)

            G_logits = tap(G_logits, "G_logits", self.enable_taps)

        else:
            # Original prediction method with normalization stack
            # Apply input normalization (gated behind flag)
            # Massive LayerNorm stack to normalize
            # The comment to the right is the number of steps to grok for ops: mul, add
            # all seed 33232323
            if not self.use_dense_features:
                head_input = self.input_norm(head_input)  # inf, inf
            head_input = self.extra_norm1(head_input)  # inf, 75
            head_input = self.extra_norm2(head_input)  # 1119, 51
            head_input = self.extra_norm3(head_input)  # 649, 52
            head_input = self.extra_norm4(head_input)  # 157, 46
            head_input = self.extra_norm5(head_input)  # 108, 50
            head_input = self.extra_norm6(head_input)  # 216, 46
            head_input = self.extra_norm7(head_input)  # 108, 50
            head_input = self.extra_norm8(head_input)  # 216, 46
            head_input = self.extra_norm9(head_input)  # 108, 50
            head_input = self.extra_norm10(head_input)  # 216, 46

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

            out_logits = self.output_selector_head(head_input).to(dtype)

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
                pattern[0] = 1.0   # First input: positive
                pattern[1] = -1.0  # Second input: negative
                # Remaining positions stay at 0
            
            # Apply pattern to all DAG steps and all batch elements
            O = pattern.unsqueeze(0).unsqueeze(0).expand(B, self.dag_depth, -1)
        elif self.freeze_O_mul:
            # Freeze to multiplication pattern: [1, 1, 0, 0, 0, ...] for all steps
            pattern = torch.zeros(self.total_nodes, device=O.device, dtype=O.dtype)
            if self.num_initial_nodes >= 2:
                pattern[0] = 1.0   # First input: positive
                pattern[1] = 1.0   # Second input: positive
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

        if not self.training:
            # Only apply eval discretization if STEs are not already handling it
            G = (G > 0.5).to(G.dtype)
            O = torch.round(O).clamp(-1.0, 1.0)

        return O, G, out_logits

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

    def _compute_aggregates(
        self,
        working_mag: torch.Tensor,
        working_sign: torch.Tensor,
        O_step: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Aggregate linear/log terms for complex domain mixing."""
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

        # We extract the linear sign based on result of the operation
        linear_sign = torch.tanh(R_lin)

        # Map working_sign to angles for smooth sign
        w = torch.abs(O_step)
        neg_frac = 0.5 * (1.0 - working_sign)  # 1 for −1, 0 for +1, fractional if soft
        m = torch.sum(w * neg_frac, dim=-1, keepdim=True)

        # Smooth parity: +1 for even m, −1 for odd m, smooth in between when weights are fractional
        log_sign = torch.cos(math.pi * m)
        s = G_step * linear_sign + (1.0 - G_step) * log_sign

        return s

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
        m_log = l_log + G_step * delta
        m_log = torch.clamp(m_log, min=-self._log_lim, max=self._log_lim)
        # This should be clamped to not be too large
        V_mag = torch.exp(m_log)
        return V_mag

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
            O, G, out_logits = self.predict_dag_weights(input, device, dtype, B)
        else:
            # Use manually set weights directly (for testing)
            O_mag = self.test_O_mag
            O_sign = self.test_O_sign
            O = O_sign * O_mag
            G = self.test_G
            out_logits = self.test_out_logits

        # Save the weights state after any hardening for logging
        # Separate tracking for training vs eval states
        if self.training:
            self._last_train_G = G.detach()
            self._last_train_O = O.detach()
        else:
            self._last_eval_G = G.detach()
            self._last_eval_O = O.detach()

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

        for step in range(self.dag_depth):
            O_step = O[:, step, :]
            G_step = G[:, step].unsqueeze(-1)

            if self.use_simple_domain_mixing:
                # Use simple domain mixing approach from working version
                R_mixed = self._compute_simple_domain_mixed_result(
                    working_mag, working_sign, O_step, G_step
                )
                R_mixed = tap(R_mixed, "R_mixed", self.enable_taps)

                # Debug: save for compatibility
                self._debug_R_lin.append(R_mixed.clone())
                self._debug_R_log.append(R_mixed.clone())

                if self._is_nan("R_mixed", R_mixed) and self.training:
                    pdb.set_trace()

                # For simple mixing, use R_mixed for both sign and magnitude computation
                # Simple sign computation with sharp tanh
                linear_sign = torch.tanh(R_mixed / 1e-4)

                # Use complex sign computation with cos trick from the original domain mixing
                w = torch.abs(O_step)
                neg_frac = 0.5 * (
                    1.0 - working_sign
                )  # 1 for −1, 0 for +1, fractional if soft
                m = torch.sum(w * neg_frac, dim=-1, keepdim=True)
                # Smooth parity: +1 for even m, −1 for odd m, smooth in between when weights are fractional
                log_sign = torch.cos(math.pi * m)

                V_sign_new = G_step * linear_sign + (1.0 - G_step) * log_sign

                # Simple magnitude computation
                linear_mag = torch.clamp(torch.abs(R_mixed), max=self._mag_max)
                R_mixed_clamped = self.soft_clamp(
                    R_mixed, min=-self._log_lim, max=self._log_lim
                )
                log_mag_result = torch.exp(R_mixed_clamped)
                V_mag_new = G_step * linear_mag + (1.0 - G_step) * log_mag_result
            else:
                # Original complex domain mixing
                R_lin, R_log = self._compute_aggregates(
                    working_mag, working_sign, O_step
                )
                R_lin = tap(R_lin, "R_lin", self.enable_taps)
                R_log = tap(R_log, "R_log", self.enable_taps)

                # Debug: save intermediate computation results
                self._debug_R_lin.append(R_lin.clone())
                self._debug_R_log.append(R_log.clone())

                if self._is_nan("R_lin", R_lin) and self.training:
                    pdb.set_trace()
                if self._is_nan("R_log", R_log) and self.training:
                    pdb.set_trace()

                V_sign_new = self._compute_new_sign(R_lin, working_sign, O_step, G_step)
                V_mag_new = self._compute_new_magnitude(R_lin, R_log, G_step)

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
                print(f"R_mixed: {R_mixed[0].item():.6f}")

                # Check for numerical issues
                if torch.abs(R_mixed).max() > self._log_lim:
                    print(f"⚠️  R_mixed out of bounds!")
                if torch.abs(log_mag_result).max() > self._mag_max:
                    print(f"⚠️  log_mag_result out of bounds!")

                print(f"Linear sign: {linear_sign[0].item():.6f}")
                print(f"Log sign: {log_sign[0].item():.6f}")
                print(f"Final V_sign: {V_sign_new[0].item():.6f}")
                print(f"Linear mag: {linear_mag[0].item():.6f}")
                print(f"Log mag result: {log_mag_result[0].item():.6f}")
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
