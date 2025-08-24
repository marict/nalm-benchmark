#!/usr/bin/env python3
"""
Test soft clamping functionality in DAG layer.
"""

import torch

from stable_nalu.layer.dag import DAGLayer


def test_soft_clamping_parameter_exists():
    """Test that use_soft_clamping parameter is properly stored."""
    layer_hard = DAGLayer(2, 1, 1, use_soft_clamping=False)
    layer_soft = DAGLayer(2, 1, 1, use_soft_clamping=True)

    assert layer_hard.use_soft_clamping is False
    assert layer_soft.use_soft_clamping is True
    print("✓ use_soft_clamping parameter properly stored")


def test_soft_clamping_identical_weights():
    """Test that soft and hard clamping produce identical results with same weights when not hitting boundaries."""
    torch.manual_seed(42)

    # Create layers with identical weights
    layer_hard = DAGLayer(2, 1, 1, use_soft_clamping=False, _enable_taps=False)
    layer_soft = DAGLayer(2, 1, 1, use_soft_clamping=True, _enable_taps=False)
    layer_soft.load_state_dict(layer_hard.state_dict())

    layer_hard.eval()
    layer_soft.eval()

    x = torch.randn(1, 2)

    with torch.no_grad():
        out_hard = layer_hard(x)
        out_soft = layer_soft(x)

    # Should be identical when not hitting clamp boundaries
    difference = abs(out_hard.item() - out_soft.item())
    assert difference < 1e-7, f"Expected identical results, got difference {difference}"
    print(f"✓ Identical weights produce identical results (diff: {difference:.2e})")


def test_soft_clamping_extreme_values():
    """Test soft clamping with extreme values that may trigger boundary conditions."""
    torch.manual_seed(42)

    # Create layers with identical weights
    layer_hard = DAGLayer(2, 1, 1, use_soft_clamping=False, _enable_taps=False)
    layer_soft = DAGLayer(2, 1, 1, use_soft_clamping=True, _enable_taps=False)
    layer_soft.load_state_dict(layer_hard.state_dict())

    layer_hard.eval()
    layer_soft.eval()

    # Test with extreme values that might trigger clamping
    test_cases = [
        torch.tensor([[1000.0, -1000.0]]),  # Very large values
        torch.tensor([[1e-6, 1e6]]),  # Mixed small/large
        torch.tensor([[-1e6, -1e6]]),  # Large negative values
    ]

    for i, x in enumerate(test_cases):
        with torch.no_grad():
            out_hard = layer_hard(x)
            out_soft = layer_soft(x)

        difference = abs(out_hard.item() - out_soft.item())
        print(f"✓ Test case {i+1}: difference = {difference:.2e}")

        # Both should produce finite results
        assert torch.isfinite(
            out_hard
        ), f"Hard clamping produced non-finite result: {out_hard.item()}"
        assert torch.isfinite(
            out_soft
        ), f"Soft clamping produced non-finite result: {out_soft.item()}"


def test_soft_clamping_gradient_flow():
    """Test that soft clamping allows gradient flow in training mode."""
    torch.manual_seed(42)

    layer_hard = DAGLayer(2, 1, 1, use_soft_clamping=False, _enable_taps=False)
    layer_soft = DAGLayer(2, 1, 1, use_soft_clamping=True, _enable_taps=False)
    layer_soft.load_state_dict(layer_hard.state_dict())

    layer_hard.train()
    layer_soft.train()

    x = torch.randn(1, 2, requires_grad=True)
    target = torch.randn(1, 1)

    # Test gradient flow for hard clamping
    out_hard = layer_hard(x)
    loss_hard = torch.nn.functional.mse_loss(out_hard, target)
    loss_hard.backward()
    grad_hard = x.grad.clone()
    x.grad.zero_()

    # Test gradient flow for soft clamping
    out_soft = layer_soft(x)
    loss_soft = torch.nn.functional.mse_loss(out_soft, target)
    loss_soft.backward()
    grad_soft = x.grad.clone()

    # Both should have gradients
    assert torch.any(grad_hard != 0), "Hard clamping should produce non-zero gradients"
    assert torch.any(grad_soft != 0), "Soft clamping should produce non-zero gradients"
    print("✓ Both hard and soft clamping allow gradient flow")


def test_soft_clamping_with_domain_mixing():
    """Test soft clamping compatibility with different domain mixing modes."""
    torch.manual_seed(42)
    x = torch.randn(1, 2)

    # Test combinations
    configs = [
        (True, False),  # Simple domain, hard clamp
        (True, True),  # Simple domain, soft clamp
        (False, False),  # Complex domain, hard clamp
        (False, True),  # Complex domain, soft clamp
    ]

    for use_simple, use_soft in configs:
        layer = DAGLayer(
            2,
            1,
            1,
            use_simple_domain_mixing=use_simple,
            use_soft_clamping=use_soft,
            _enable_taps=False,
        )
        layer.eval()

        with torch.no_grad():
            out = layer(x)

        assert torch.isfinite(
            out
        ), f"Config (simple={use_simple}, soft={use_soft}) produced non-finite result"
        print(
            f"✓ Compatible with simple_domain_mixing={use_simple}, soft_clamping={use_soft}"
        )


def main():
    """Run all soft clamping tests."""
    print("Testing Soft Clamping Functionality")
    print("=" * 40)

    test_soft_clamping_parameter_exists()
    test_soft_clamping_identical_weights()
    test_soft_clamping_extreme_values()
    test_soft_clamping_gradient_flow()
    test_soft_clamping_with_domain_mixing()

    print("=" * 40)
    print("✅ All soft clamping tests passed!")


if __name__ == "__main__":
    main()
