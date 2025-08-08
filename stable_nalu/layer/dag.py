from __future__ import annotations

import torch
import torch.nn as nn

from ..abstract import ExtendedTorchModule

"""
grokking commands:

python3 experiments/single_layer_benchmark.py --no-cuda --layer-type DAG --operation add --input-size 3 --batch-size 1000 --max-iterations 300000 --log-interval 1000 --clip-grad-norm 1.0


python3 experiments/single_layer_benchmark.py --no-cuda --layer-type DAG --operation mul --input-size 3 --batch-size 1000 --max-iterations 300000 --log-interval 1000 --clip-grad-norm 1.0

Things to try (on cloud):
python /Users/paul_curry/ai2/runpod_service/runpod_service.py experiments/single_layer_benchmark.py --layer-type DAG --operation sub --input-size 3 --batch-size 10000 --max-iterations 300000 --log-interval 1000 --clip-grad-norm 1.0 --pod-name nalm-sub

python /Users/paul_curry/ai2/runpod_service/runpod_service.py experiments/single_layer_benchmark.py --layer-type DAG --operation div --input-size 3 --batch-size 100000 --max-iterations 300000 --log-interval 1000 --clip-grad-norm 1.0 --pod-name nalm-div


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
        writer=None,
        name: str | None = None,
        dag_depth: int | None = None,
        freeze_g_linear: bool = False,
        freeze_g_log: bool = False,
        use_ste: bool = False,
        use_layer_norm: bool = True,
        use_attention: bool = True,
        attn_dim: int = 64,
        attn_layers: int = 1,
        dropout_p: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__("dag", writer=writer, name=name, **kwargs)

        if out_features != 1:
            raise ValueError(
                f"DAGLayer currently supports out_features == 1, got {out_features}"
            )

        self.in_features = in_features
        self.out_features = out_features

        # Execution graph sizing: initial nodes come from input features
        # Total nodes = initial nodes + dag_depth (each step produces one new node)
        self.dag_depth = (
            int(dag_depth) if dag_depth is not None else max(1, in_features - 1)
        )
        self.num_initial_nodes = in_features
        self.total_nodes = self.num_initial_nodes + self.dag_depth

        self.freeze_g_linear = bool(freeze_g_linear)
        self.freeze_g_log = bool(freeze_g_log)
        self.use_ste = bool(use_ste)
        self.use_attention = bool(use_attention)

        # Normalize inputs and optionally enhance with a small MLP or attention + dropout
        self.input_norm = nn.LayerNorm(in_features) if use_layer_norm else None
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_p)

        # Optional self-attention pre-head over tokens with positional embeddings
        if self.use_attention:
            self.attn_dim = int(attn_dim)
            self.attn_layers = int(attn_layers)
            # Learned positional embeddings for positions [0..in_features-1]
            self.pos_embed = nn.Parameter(torch.zeros(in_features, self.attn_dim))
            # Project scalar token -> model dim and back
            self.token_in = nn.Linear(1, self.attn_dim)
            self.token_out = nn.Linear(self.attn_dim, 1)
            # Single-head self-attention blocks
            self.attn_blocks = nn.ModuleList(
                [
                    nn.ModuleDict(
                        {
                            "mha": nn.MultiheadAttention(
                                embed_dim=self.attn_dim,
                                num_heads=1,
                                dropout=dropout_p,
                                batch_first=True,
                            ),
                            "ln1": nn.LayerNorm(self.attn_dim),
                            "ff1": nn.Linear(self.attn_dim, self.attn_dim),
                            "ff2": nn.Linear(self.attn_dim, self.attn_dim),
                            "ln2": nn.LayerNorm(self.attn_dim),
                            "act": nn.GELU(),
                            "drop": nn.Dropout(dropout_p),
                        }
                    )
                    for _ in range(self.attn_layers)
                ]
            )

        # Small prediction heads mapping input -> structure
        # Operand selector matrix O: shape (dag_depth, total_nodes)
        self.O_head = nn.Linear(in_features, self.dag_depth * self.total_nodes)
        # Domain gate G in [0,1] per step: shape (dag_depth,)
        self.G_head = nn.Linear(in_features, self.dag_depth)

        # Initialize heads similar to standard small heads
        nn.init.normal_(self.O_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.O_head.bias)
        nn.init.normal_(self.G_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.G_head.bias)

        # Numerical guards
        self._mag_min = 1e-6
        self._mag_max = 1e28
        self._log_lim = 100.0

    def reset_parameters(self) -> None:
        # Reinitialize prediction heads
        nn.init.normal_(self.O_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.O_head.bias)
        nn.init.normal_(self.G_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.G_head.bias)

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
        # Blend between linear magnitude and exp(log-magnitude)
        linear_mag = torch.clamp(torch.abs(R_mag), max=self._mag_max)
        R_mag_clamped = torch.clamp(R_mag, min=-self._log_lim, max=self._log_lim)
        log_mag_result = torch.exp(R_mag_clamped)
        return G_step * linear_mag + (1.0 - G_step) * log_mag_result

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input: (B, in_features)
        if input.dim() != 2 or input.size(1) != self.in_features:
            raise ValueError(
                f"Expected input of shape (B, {self.in_features}), got {tuple(input.shape)}"
            )

        device = input.device
        dtype = torch.float64 if device.type != "mps" else torch.float32
        B = input.size(0)

        # Initial node magnitudes and signs come directly from input features
        init_mag = torch.clamp(input.abs(), min=self._mag_min, max=self._mag_max).to(
            dtype
        )
        init_sign = torch.where(
            input >= 0,
            torch.tensor(1.0, device=device),
            torch.tensor(-1.0, device=device),
        ).to(dtype)

        # Predict structure from input
        head_input = (
            self.input_norm(input) if self.input_norm is not None else input
        )  # (B, in_features)
        if self.use_attention:
            # Build token sequence (B, N, D)
            tokens = input.unsqueeze(-1)  # (B, N, 1)
            tokens = self.token_in(tokens)  # (B, N, D)
            tokens = tokens + self.pos_embed.unsqueeze(
                0
            )  # add learned positional encodings
            for blk in self.attn_blocks:
                # Self-attention + residual + norm
                resid = tokens
                attn_out, _ = blk["mha"](tokens, tokens, tokens, need_weights=False)
                tokens = blk["ln1"](resid + blk["drop"](attn_out))
                # Feed-forward + residual + norm
                resid = tokens
                f = blk["act"](blk["ff1"](tokens))
                f = blk["drop"](blk["ff2"](f))
                tokens = blk["ln2"](resid + f)
            head_input = self.token_out(tokens).squeeze(-1)  # (B, N)

        O_flat = self.O_head(head_input)  # (B, dag_depth * total_nodes)
        O = O_flat.view(B, self.dag_depth, self.total_nodes).to(dtype)
        G = torch.sigmoid(self.G_head(head_input)).to(dtype)  # (B, dag_depth)

        # Optionally freeze G to linear domain (G==1)
        if self.freeze_g_linear:
            G = torch.ones_like(G)
        if self.freeze_g_log:
            G = torch.zeros_like(G)

        # Optional STE discretisation in training for stability/inductive bias
        if self.training:
            if self.use_ste:
                O = self._ste_round(O)
                # Hard bound O to represent inclusion/exclusion/sign only
                O = torch.clamp(O, -1.0, 1.0)
                G_hard = (G > 0.5).to(G.dtype)
                G = G_hard + (G - G.detach())
            else:
                # Train with continuous bounded coefficients (no rounding)
                O = torch.tanh(O)
        else:
            # Eval: use hard forward for O and G
            O = torch.round(O)
            O = torch.clamp(O, -1.0, 1.0)
            G = (G > 0.5).to(G.dtype)

        # Prepare working tensors: start with initial nodes, append one per step
        working_mag = torch.zeros(B, self.total_nodes, dtype=dtype, device=device)
        working_sign = torch.zeros(B, self.total_nodes, dtype=dtype, device=device)
        working_mag[:, : self.num_initial_nodes] = init_mag
        working_sign[:, : self.num_initial_nodes] = init_sign

        # Execute DAG steps
        for step in range(self.dag_depth):
            O_step = O[:, step, :]  # (B, total_nodes)
            G_step = G[:, step].unsqueeze(-1)  # (B, 1)

            # Causal mask: only allow using already-computed nodes
            valid_nodes = self.num_initial_nodes + step
            causal = torch.zeros_like(O_step)
            causal[:, :valid_nodes] = 1.0
            O_step = O_step * causal

            R_mag = self._compute_domain_mixed_result(
                working_mag, working_sign, O_step, G_step
            )
            V_sign_new = self._compute_new_sign(R_mag, working_sign, O_step, G_step)
            V_mag_new = self._compute_new_magnitude(R_mag, G_step)

            V_mag_new = torch.clamp(V_mag_new, min=self._mag_min, max=self._mag_max)
            V_sign_new = torch.clamp(V_sign_new, min=-1.0, max=1.0)

            idx = self.num_initial_nodes + step
            # Avoid in-place assignment that breaks autograd by using scatter to produce new tensors
            index_tensor = torch.full((B, 1), idx, device=device, dtype=torch.long)
            working_mag = working_mag.scatter(-1, index_tensor, V_mag_new)
            working_sign = working_sign.scatter(-1, index_tensor, V_sign_new)

        # Final output is the last node
        final_idx = self.total_nodes - 1
        final_value = working_sign[:, final_idx] * working_mag[:, final_idx]

        if final_value.isnan().any():
            print(f"NAN in final value: {final_value}")
            print(f"Working mag: {working_mag}")
            raise ValueError("NAN in final value")

        # Return with expected dtype
        return final_value.to(input.dtype).unsqueeze(-1)  # (B, 1)
