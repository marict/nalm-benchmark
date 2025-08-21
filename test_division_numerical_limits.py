#!/usr/bin/env python3
"""
Test different numerical limits for division training.
The mag_min, mag_max, and log_lim_offset parameters might be critical for division stability.
"""

import sys
import time

import torch

from stable_nalu.layer.dag import DAGLayer


def test_division_with_limits(
    mag_min, mag_max, log_lim_offset, test_name, iterations=500
):
    """Test division with specific numerical limits."""
    print(f"\n🔧 Testing {test_name}")
    print(
        f"   mag_min={mag_min:.0e}, mag_max={mag_max:.0e}, log_lim_offset={log_lim_offset}"
    )

    # Create layer with custom limits
    layer = DAGLayer(
        in_features=2,
        out_features=1,
        dag_depth=3,
        mag_min=mag_min,
        mag_max=mag_max,
        log_lim_offset=log_lim_offset,
        enable_taps=False,
    )

    # Simple training setup
    torch.manual_seed(42)  # Fixed seed for comparison

    # Create simple division data
    batch_size = 16
    x1 = torch.randn(batch_size, 1) * 2 + 0.1  # Avoid exact zeros
    x2 = torch.randn(batch_size, 1) * 2 + 0.1
    inputs = torch.cat([x1, x2], dim=1)
    targets = (x1 / x2).squeeze()

    # Simple optimizer
    optimizer = torch.optim.Adam(layer.parameters(), lr=1e-3)

    start_time = time.time()
    initial_loss = None
    final_loss = None

    for i in range(iterations):
        optimizer.zero_grad()
        outputs = layer(inputs).squeeze()
        loss = torch.nn.functional.mse_loss(outputs, targets)

        if i == 0:
            initial_loss = loss.item()

        loss.backward()

        # Check for gradient issues
        total_grad_norm = 0
        for param in layer.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.norm().item() ** 2
        total_grad_norm = total_grad_norm**0.5

        # Clip gradients if they explode
        torch.nn.utils.clip_grad_norm_(layer.parameters(), 1.0)

        optimizer.step()

        if i == iterations - 1:
            final_loss = loss.item()

        # Early termination if loss explodes
        if not torch.isfinite(loss) or loss.item() > 1e6:
            print(f"   ❌ DIVERGED at step {i}, loss={loss.item():.2e}")
            return {
                "success": False,
                "error": "diverged",
                "initial_loss": initial_loss,
                "final_loss": loss.item(),
                "duration": time.time() - start_time,
            }

    duration = time.time() - start_time
    improvement = initial_loss / final_loss if final_loss > 0 else float("inf")

    print(f"   Initial loss: {initial_loss:.2e}")
    print(f"   Final loss:   {final_loss:.2e}")
    print(f"   Improvement:  {improvement:.1f}x")
    print(f"   Duration:     {duration:.1f}s")

    # Check if this is a good result
    success = final_loss < 0.1 * initial_loss and final_loss < 100
    status = "✅ GOOD" if success else "❌ POOR"
    print(f"   Status: {status}")

    return {
        "success": success,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "improvement": improvement,
        "duration": duration,
    }


def main():
    """Test different numerical limit configurations."""
    print("Division Numerical Limits Experiment")
    print("=" * 45)

    # Test different numerical ranges
    # Current defaults: mag_min=1e-11, mag_max=1e6, log_lim_offset=1.0

    test_configs = [
        # Baseline (current defaults)
        (1e-11, 1e6, 1.0, "baseline"),
        # Tighter magnitude limits
        (1e-8, 1e4, 1.0, "tighter_mag_range"),
        (1e-6, 1e3, 1.0, "much_tighter_mag"),
        # Looser magnitude limits
        (1e-12, 1e8, 1.0, "looser_mag_range"),
        (1e-15, 1e10, 1.0, "much_looser_mag"),
        # Different log limits
        (1e-11, 1e6, 0.5, "smaller_log_offset"),
        (1e-11, 1e6, 2.0, "larger_log_offset"),
        (1e-11, 1e6, 0.1, "tiny_log_offset"),
        # Division-friendly: tighter control for numerical stability
        (1e-9, 1e5, 0.8, "division_optimized"),
        (1e-8, 1e4, 0.5, "division_conservative"),
        # Very tight for maximum stability
        (1e-6, 1e2, 0.3, "ultra_conservative"),
    ]

    results = []

    print(
        f"Testing {len(test_configs)} configurations with {500} training steps each..."
    )

    for mag_min, mag_max, log_lim_offset, name in test_configs:
        try:
            result = test_division_with_limits(mag_min, mag_max, log_lim_offset, name)
            results.append((name, result))
        except Exception as e:
            print(f"   ❌ ERROR: {str(e)}")
            results.append((name, {"success": False, "error": str(e)}))

    # Analysis
    print(f"\n📊 RESULTS SUMMARY:")
    print(f"=" * 45)

    successful_configs = [
        (name, res) for name, res in results if res.get("success", False)
    ]

    if successful_configs:
        print(f"🎉 Successful configurations:")

        # Sort by improvement ratio
        successful_configs.sort(key=lambda x: x[1]["improvement"], reverse=True)

        for i, (name, res) in enumerate(successful_configs[:5]):  # Top 5
            improvement = res["improvement"]
            final_loss = res["final_loss"]
            print(
                f"   {i+1}. {name}: {improvement:.1f}x improvement, final_loss={final_loss:.2e}"
            )

        # Find best overall
        best_name, best_res = successful_configs[0]
        print(f"\n🏆 BEST: {best_name}")
        print(f"   Final loss: {best_res['final_loss']:.2e}")
        print(f"   Improvement: {best_res['improvement']:.1f}x")

        # Get the config details
        best_config = next(config for config in test_configs if config[3] == best_name)
        mag_min, mag_max, log_lim_offset, _ = best_config
        print(
            f"   Parameters: mag_min={mag_min:.0e}, mag_max={mag_max:.0e}, log_lim_offset={log_lim_offset}"
        )

        return True

    else:
        print(f"❌ No configuration achieved good results.")
        print(f"   All configs either diverged or failed to improve significantly.")

        # Show least bad results
        valid_results = [(name, res) for name, res in results if "final_loss" in res]
        if valid_results:
            valid_results.sort(key=lambda x: x[1]["final_loss"])
            print(f"\n🔍 Least bad results:")
            for name, res in valid_results[:3]:
                print(f"   {name}: final_loss={res['final_loss']:.2e}")

        return False


if __name__ == "__main__":
    success = main()

    if success:
        print(f"\n🎉 SUCCESS: Found numerical limits that improve division training!")
        print(f"   Consider using these parameters for division experiments.")
    else:
        print(
            f"\n🔬 No breakthrough found, but may have identified better numerical ranges."
        )

    sys.exit(0 if success else 1)
