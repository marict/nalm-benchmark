import os
import sys
import typing

import numpy as np
import torch

# Add parent directory to path to find runpod_service
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import runpod_service.wandb_setup as wandb


class TapContext:
    """Global context manager for tap logging."""

    def __init__(self):
        self.epoch_i = 0

    def set_epoch_i(self, epoch_i: int):
        self.epoch_i = epoch_i

    def get_epoch_i(self):
        return self.epoch_i


# Global instance
tap_context = TapContext()


class GradTap(torch.autograd.Function):
    """Identity op that logs tensor stats in forward and backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, tag: str) -> torch.Tensor:
        """Pass-through that logs forward stats."""
        ctx.tag = tag
        with torch.no_grad():
            t = x.detach()
            wandb.wrapper.log(
                {
                    f"fw/{tag}/min": float(torch.nan_to_num(t).min()),
                    f"fw/{tag}/max": float(torch.nan_to_num(t).max()),
                    f"fw/{tag}/mean": float(torch.nan_to_num(t).mean()),
                    f"fw/{tag}/std": float(torch.nan_to_num(t).std()),
                },
                step=tap_context.get_epoch_i(),
                commit=False,
            )
        return x

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> typing.Tuple[torch.Tensor, None]:
        """Pass-through that logs backward grad stats."""
        tag = ctx.tag
        g = grad_out
        g_safe = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
        wandb.wrapper.log(
            {
                f"bw/{tag}/min": float(g_safe.min()),
                f"bw/{tag}/max": float(g_safe.max()),
                f"bw/{tag}/mean": float(g_safe.mean()),
                f"bw/{tag}/std": float(g_safe.std()),
            },
            step=tap_context.get_epoch_i(),
            commit=False,
        )
        return grad_out, None


def tap(x: torch.Tensor, tag: str, enable: bool = True) -> torch.Tensor:
    """Convenience wrapper for GradTap."""
    if enable:
        return GradTap.apply(x, tag)
    return x
