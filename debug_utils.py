import typing

import runpod_service.wandb_setup as wandb
import torch


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
                    f"fw/{tag}/finite_frac": float(torch.isfinite(t).float().mean()),
                }
            )
        return x.clone()

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
                f"bw/{tag}/finite_frac": float(torch.isfinite(g).float().mean()),
            }
        )
        return grad_out, None


def tap(x: torch.Tensor, tag: str, enable: bool = True) -> torch.Tensor:
    """Convenience wrapper for GradTap."""
    if enable:
        return GradTap.apply(x, tag)
    return x
