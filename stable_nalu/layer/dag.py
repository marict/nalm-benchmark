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
        writer: str = None,
        name: str | None = None,
        _enable_debug_logging: bool = False,
        _enable_taps: bool = False,
        _do_not_predict_weights: bool = False,
        op: str = None,
        freeze_G: bool = False,
        freeze_O: bool = False,
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

        # Force in_features=2 for single arithmetic operations
        if in_features != 2:
            raise ValueError(f"Simplified DAGLayer only supports in_features=2, got {in_features}")
            
        self.in_features = in_features
        self.out_features = out_features

        # Fixed architecture: depth=1, 2 inputs, 1 output
        self.num_initial_nodes = 2
        
        self.op = op
        self.freeze_G = bool(freeze_G)
        self.freeze_O = bool(freeze_O)
        self.enable_debug_logging = bool(_enable_debug_logging)
        self.enable_taps = bool(_enable_taps)
        self._do_not_predict_weights = bool(_do_not_predict_weights)
        self.unfreeze_eval = bool(unfreeze_eval)
        self.G_perturbation = float(G_perturbation)
        self.freeze_input_norm = bool(freeze_input_norm)
        self.use_norm = bool(use_norm)
        self.single_G = bool(single_G)
        self.epsilon_smooth = bool(epsilon_smooth)

        # Replace neural heads with simple learnable parameters
        # O parameters: [weight_for_input0, weight_for_input1] 
        self.O_params = nn.Parameter(torch.tensor([1.0, 1.0]))  # Default to [1, 1] for add/mul
        
        # G parameter: single scalar for domain mixing (if single_G) or 2 scalars (if dual_G)
        if self.single_G:
            self.G_param = nn.Parameter(torch.tensor(0.0))  # Single scalar, sigmoid -> [0,1]
        else:
            self.G_params = nn.Parameter(torch.tensor([0.0, 0.0]))  # [G_lin_logit, G_log_logit]

        # Optional input normalization (much simpler now)  
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

        self.reset_parameters()

        self._mag_min = 1e-11
        self._mag_max = 1e6
        self._log_lim = math.log(self._mag_max) - 1.0

    def reset_parameters(self) -> None:
        # Initialize O parameters based on operation
        with torch.no_grad():
            if self.op in ["add", "mul"]:
                self.O_params.data = torch.tensor([1.0, 1.0])  # [1, 1] for add/mul
            elif self.op in ["sub", "div"]:  
                self.O_params.data = torch.tensor([1.0, -1.0])  # [1, -1] for sub/div
            else:
                # Default initialization for unknown operations
                self.O_params.data = torch.tensor([1.0, 1.0])
                
        # Initialize G parameters to neutral (0 logits = equal probability)
        with torch.no_grad():
            if self.single_G:
                self.G_param.data.zero_()  # 0 logit -> sigmoid(0) = 0.5
            else:
                self.G_params.data.zero_()  # [0, 0] -> equal softmax probabilities

    def predict_dag_weights(self, input: torch.Tensor, device, dtype, B: int):
        """Generate O and G weights using simple learnable parameters (input-independent)."""
        
        # O weights: simply broadcast our learned parameters to batch size
        # Shape: [B, 2] (batch_size, 2_inputs)
        O = self.O_params.unsqueeze(0).expand(B, -1)  # [B, 2]
        
        # G weights: compute from our learned parameters  
        if self.single_G:
            # Single G: apply sigmoid to get value in [0, 1]
            G_single = torch.sigmoid(self.G_param)  # scalar
            
            if self.epsilon_smooth:
                # Apply epsilon smoothing for training stability
                eps = 1e-5
                G_single = eps + (1.0 - 2.0 * eps) * G_single  # Map [0,1] to [1e-5, 1-1e-5]
            
            G_lin = G_single.unsqueeze(0).expand(B)  # [B] 
            G_log = 1.0 - G_single.unsqueeze(0).expand(B)  # [B]
        else:
            # Dual G: apply softmax to get probabilities
            G_probs = torch.softmax(self.G_params, dim=0)  # [2] -> [G_lin_prob, G_log_prob]
            G_lin = G_probs[0].unsqueeze(0).expand(B)  # [B]
            G_log = G_probs[1].unsqueeze(0).expand(B)  # [B]

        # Apply frozen O values if specified (override learned values)
        if self.freeze_O:
            if self.op in ["add", "mul"]:
                O = torch.tensor([1.0, 1.0], device=device, dtype=dtype).unsqueeze(0).expand(B, -1)
            elif self.op in ["sub", "div"]:
                O = torch.tensor([1.0, -1.0], device=device, dtype=dtype).unsqueeze(0).expand(B, -1)
            else:
                O = torch.tensor([1.0, 1.0], device=device, dtype=dtype).unsqueeze(0).expand(B, -1)

        # Apply frozen G values if specified (override learned values)
        if self.freeze_G:
            if self.op in ["add", "sub"]:
                # Linear operations: G=1.0 for linear, G=0.0 for log
                G_lin = torch.ones(B, device=device, dtype=dtype) 
                G_log = torch.zeros(B, device=device, dtype=dtype)
            elif self.op in ["mul", "div"]:
                # Log operations: G=0.0 for linear, G=1.0 for log  
                G_lin = torch.zeros(B, device=device, dtype=dtype)
                G_log = torch.ones(B, device=device, dtype=dtype)
            else:
                # Default: equal probabilities
                G_lin = torch.full((B,), 0.5, device=device, dtype=dtype)
                G_log = torch.full((B,), 0.5, device=device, dtype=dtype)

        return O, G_lin, G_log

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
        """Simplified forward pass for single arithmetic operations (depth=1)."""
        if input.dim() != 2 or input.size(1) != self.in_features:
            raise ValueError(
                f"Expected input of shape (B, {self.in_features}), got {tuple(input.shape)}"
            )

        device = input.device
        dtype = input.dtype  # Use input's dtype for consistency
        B = input.size(0)

        # Apply input normalization if enabled
        if self.input_norm is not None:
            input = self.input_norm(input)

        # Convert input to computation dtype
        input = input.to(dtype)

        # Get operation weights and domain mixing parameters
        O, G_lin, G_log = self.predict_dag_weights(input, device, dtype, B)

        # Extract input values
        x1 = input[:, 0]  # [B]
        x2 = input[:, 1]  # [B]
        
        # Apply operation weights: O[0] * x1 + O[1] * x2 (typically [1,1] or [1,-1])
        weighted_x1 = O[:, 0] * x1  # [B]
        weighted_x2 = O[:, 1] * x2  # [B]

        # Always compute both linear and log domain results
        # Linear domain: weighted sum (works for add/sub with correct O weights)
        linear_result = weighted_x1 + weighted_x2  # [B]
        
        # Log domain: multiplicative operations in log space
        abs_x1 = torch.abs(weighted_x1)
        abs_x2 = torch.abs(weighted_x2)
        log_sum = torch.log(torch.clamp(abs_x1, min=self._mag_min)) + torch.log(torch.clamp(abs_x2, min=self._mag_min))
        
        # Handle signs for log domain
        sign1 = torch.sign(weighted_x1)
        sign2 = torch.sign(weighted_x2)
        log_sign = sign1 * sign2
        log_result = log_sign * torch.exp(torch.clamp(log_sum, min=-self._log_lim, max=self._log_lim))  # [B]
        
        # Mix domains using learned G parameters
        # For add/sub: G should learn G_lin→1, G_log→0 
        # For mul/div: G should learn G_lin→0, G_log→1
        mixed_result = G_lin * linear_result + G_log * log_result

        return mixed_result.unsqueeze(-1)  # [B, 1] for compatibility

    def calculate_sparsity_error(self, operation: str) -> float:
        """Calculate sparsity error based on G (gating) weights.

        Args:
            operation: The arithmetic operation (kept for API compatibility, not used)

        Returns:
            sparsity_error: min(|G|, |1-G|) measuring distance from G to discrete values {0,1}
        """
        # Get the current raw G weights directly from parameters
        if self.single_G:
            # Single G: get the sigmoid value of our parameter
            g_value = torch.sigmoid(self.G_param).item()
        else:
            # Dual G: get the softmax probabilities
            g_probs = torch.softmax(self.G_params, dim=0)
            g_value = g_probs[0].item()  # Use linear domain probability
        
        # Calculate sparsity error for G (gating parameter)
        # G should be close to 0 (log domain) or 1 (linear domain) for optimal performance
        # Sparsity error = min(|G|, |1-G|) measures distance to discrete values {0, 1}
        sparsity_error = min(g_value, 1.0 - g_value)

        return sparsity_error
