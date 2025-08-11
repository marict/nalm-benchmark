from __future__ import annotations

import torch
import torch.nn as nn

from ..abstract import ExtendedTorchModule

"""
grokking commands:

python3 experiments/single_layer_benchmark.py --no-cuda --layer-type DAG --operation add --input-size 3 --batch-size 1000 --max-iterations 300000 --log-interval 1000 --clip-grad-norm 1.0

python3 experiments/single_layer_benchmark.py --no-cuda --layer-type DAG --operation mul --input-size 3 --batch-size 256 --max-iterations 300000 --log-interval 1000 --clip-grad-norm 1.0

---

python /Users/paul_curry/ai2/runpod_service/runpod_service.py experiments/single_layer_benchmark.py --layer-type DAG --operation sub --input-size 3 --batch-size 256 --max-iterations 25000 --log-interval 100 --clip-grad-norm 1.0 --pod-name nalm-sub

python /Users/paul_curry/ai2/runpod_service/runpod_service.py experiments/single_layer_benchmark.py --layer-type DAG --operation div --input-size 3 --batch-size 256 --max-iterations 25000 --log-interval 100 --clip-grad-norm 1.0 --pod-name nalm-div


---
SELECTOR_TAU_DEFAULT

groks at: 11000
python experiments/single_layer_benchmark.py --no-cuda --layer-type DAG --operation div --input-size 3 --batch-size 1024 --max-iterations 300000 --log-interval 1000 --clip-grad-norm 1.0 --learning-rate 1e-3

groks at: 2000
python experiments/single_layer_benchmark.py --no-cuda --layer-type DAG --operation sub --input-size 3 --batch-size 1024 --max-iterations 300000 --log-interval 1000 --clip-grad-norm 1.0 --learning-rate 1e-3

groks at: 63000
python experiments/single_layer_benchmark.py --no-cuda --layer-type DAG --operation mul --input-size 3 --batch-size 1024 --max-iterations 300000 --log-interval 1000 --clip-grad-norm 1.0 --learning-rate 1e-3

groks at: 19000
python experiments/single_layer_benchmark.py --no-cuda --layer-type DAG --operation add --input-size 3 --batch-size 1024 --max-iterations 300000 --log-interval 1000 --clip-grad-norm 1.0 --learning-rate 1e-3


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

    SELECTOR_TAU_DEFAULT = 2

    def __init__(
        self,
        in_features: int,
        out_features: int,
        writer=None,
        name: str | None = None,
        dag_depth: int | None = None,
        freeze_g_linear: bool = False,
        freeze_g_log: bool = False,
        use_ste_O: bool = False,
        flip_ste_O: bool = False,
        use_ste_G: bool = True,  # Always on
        hard_eval: bool = True,
        selector_tau: float = SELECTOR_TAU_DEFAULT,
        use_output_selector: bool = True,
        **kwargs,
    ) -> None:
        super().__init__("dag", writer=writer, name=name, **kwargs)

        if out_features != 1:
            raise ValueError(
                f"DAGLayer currently supports out_features == 1, got {out_features}"
            )

        self.in_features = in_features
        self.out_features = out_features

        # Execution graph sizing: initial nodes come num subsets
        self.dag_depth = (
            int(dag_depth) if dag_depth is not None else max(1, in_features - 1)
        )
        self.num_initial_nodes = in_features
        self.total_nodes = self.num_initial_nodes + self.dag_depth

        self.freeze_g_linear = bool(freeze_g_linear)
        self.freeze_g_log = bool(freeze_g_log)
        self.use_ste_O = bool(use_ste_O)
        self.use_ste_G = bool(use_ste_G)
        self.selector_tau = float(selector_tau)
        self.flip_ste_O = bool(flip_ste_O)
        self.hard_eval = bool(hard_eval)
        self.use_output_selector = bool(use_output_selector)

        # Always-on normalizations for stability
        self.input_norm = nn.LayerNorm(in_features)
        self.head_norm = nn.LayerNorm(in_features)
        self.O_norm = nn.LayerNorm(self.total_nodes)

        # Small prediction head mapping input -> selector logits
        self.O_head = nn.Linear(in_features, self.dag_depth * self.total_nodes)
        self.O_pos_head = None
        self.O_neg_head = None

        # Domain gate G in [0,1] per step: shape (dag_depth,)
        self.G_head = nn.Linear(in_features, self.dag_depth)

        # Optional final output selector over intermediate nodes only (length == dag_depth)
        self.output_selector_head = nn.Linear(in_features, self.dag_depth)

        # Initialize heads similar to standard small heads
        nn.init.normal_(self.O_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.O_head.bias)
        # no extra heads
        nn.init.normal_(self.G_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.G_head.bias)
        nn.init.normal_(self.output_selector_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_selector_head.bias)

        # Numerical guards
        self._mag_min = 1e-6
        self._mag_max = 1e28
        self._log_lim = 100.0

    def reset_parameters(self) -> None:
        # Reinitialize prediction heads
        nn.init.normal_(self.O_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.O_head.bias)
        if self.O_pos_head is not None:
            nn.init.normal_(self.O_pos_head.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.O_pos_head.bias)
        if self.O_neg_head is not None:
            nn.init.normal_(self.O_neg_head.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.O_neg_head.bias)
        nn.init.normal_(self.G_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.G_head.bias)
        nn.init.normal_(self.output_selector_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_selector_head.bias)

    # Legacy mappings removed in simplified path

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
        if self.head_norm is not None:
            head_input = self.head_norm(head_input)
        # Guard against NaNs in features (use small positive to keep gradients alive)
        _eps = 1e-8
        head_input = torch.nan_to_num(head_input, nan=_eps, posinf=_eps, neginf=_eps)

        # Soft selector with temperature; O in [-1,1]
        O_flat = self.O_head(head_input)  # (B, dag_depth * total_nodes)
        L = O_flat.view(B, self.dag_depth, self.total_nodes)
        if self.O_norm is not None:
            L = self.O_norm(L)
        L = L.to(dtype)
        # Guard selector logits
        L = torch.nan_to_num(L, nan=_eps, posinf=_eps, neginf=_eps)
        tau = float(self.selector_tau)
        if not (tau > 0):
            tau = self.SELECTOR_TAU_DEFAULT
        sign = torch.tanh(L / tau)
        mag = torch.sigmoid(torch.abs(L) / tau)
        O = sign * mag
        O = torch.nan_to_num(O, nan=_eps, posinf=_eps, neginf=_eps)
        G_logits = self.G_head(head_input)  # (B, dag_depth)
        G_logits = torch.nan_to_num(G_logits, nan=_eps, posinf=_eps, neginf=_eps)
        G = torch.sigmoid(G_logits).to(dtype)
        G = torch.nan_to_num(G, nan=_eps, posinf=_eps, neginf=_eps)

        # Optionally freeze G to linear domain (G==1)
        if self.freeze_g_linear:
            G = torch.ones_like(G)
        if self.freeze_g_log:
            G = torch.zeros_like(G)

        # Optional STE discretisation in training for stability/inductive bias
        if self.use_ste_G:
            G_hard = (G > 0.5).to(G.dtype)
            G = G_hard + (G - G.detach())
        if self.flip_ste_O and self.use_ste_O:
            # Hard threshold on magnitude; fallback to argmax if empty
            mag_hard = (mag > 0.5).to(mag.dtype)
            empty = mag_hard.sum(dim=-1, keepdim=True) == 0
            if empty.any():
                idx = torch.argmax(L.abs(), dim=-1, keepdim=True)
                mag_hard = torch.where(
                    empty,
                    torch.zeros_like(mag_hard).scatter(-1, idx, 1.0),
                    mag_hard,
                )
            O_hard = torch.sign(L) * mag_hard
            O = O_hard + (O - O.detach())

        if self.hard_eval and not self.training:
            O = torch.round(O).clamp(-1.0, 1.0)
            G = (G > 0.5).to(G.dtype)

        # Expose selector tensors for external logging to avoid recomputation elsewhere
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

            # Causal mask: only allow using already-computed nodes
            valid_nodes = self.num_initial_nodes + step
            causal = torch.zeros_like(O_step)
            causal[:, :valid_nodes] = 1.0
            O_step = O_step * causal

            R_mag = self._compute_domain_mixed_result(
                working_mag, working_sign, O_step, G_step
            )
            R_mag = torch.nan_to_num(R_mag, nan=_eps, posinf=_eps, neginf=_eps)
            V_sign_new = self._compute_new_sign(R_mag, working_sign, O_step, G_step)
            V_mag_new = self._compute_new_magnitude(R_mag, G_step)
            V_sign_new = torch.nan_to_num(
                V_sign_new, nan=_eps, posinf=_eps, neginf=_eps
            )
            V_mag_new = torch.nan_to_num(V_mag_new, nan=_eps, posinf=_eps, neginf=_eps)

            V_mag_new = torch.clamp(V_mag_new, min=self._mag_min, max=self._mag_max)
            V_sign_new = torch.clamp(V_sign_new, min=-1.0, max=1.0)

            idx = self.num_initial_nodes + step
            # Avoid in-place assignment that breaks autograd by using scatter to produce new tensors
            index_tensor = torch.full((B, 1), idx, device=device, dtype=torch.long)
            working_mag = working_mag.scatter(-1, index_tensor, V_mag_new)
            working_sign = working_sign.scatter(-1, index_tensor, V_sign_new)

        # Final output: either last node or optional selector over intermediate nodes
        if self.use_output_selector:
            # Logits over intermediate nodes only (shape: (B, dag_depth))
            out_logits = self.output_selector_head(head_input).to(dtype)
            _eps = 1e-8
            out_logits = torch.nan_to_num(
                out_logits, nan=_eps, posinf=_eps, neginf=_eps
            )

            # Values for intermediate nodes slice [in_features : total_nodes) -> shape (B, dag_depth)
            value_vec_inter = (working_sign * working_mag)[:, self.num_initial_nodes :]
            if self.hard_eval and not self.training:
                idx = torch.argmax(
                    out_logits, dim=-1, keepdim=True
                )  # (B,1) over dag_depth
                final_value = value_vec_inter.gather(-1, idx).squeeze(-1)
            else:
                probs = torch.softmax(out_logits / tau, dim=-1)
                final_value = torch.sum(probs * value_vec_inter, dim=-1)
        else:
            # Default: use the last node
            final_idx = self.total_nodes - 1
            final_value = working_sign[:, final_idx] * working_mag[:, final_idx]

        if final_value.isnan().any():
            print(f"NAN in final value: {final_value}")
            print(f"Working mag: {working_mag}")
            raise ValueError("NAN in final value")

        # Return with expected dtype
        return final_value.to(input.dtype).unsqueeze(-1)  # (B, 1)
