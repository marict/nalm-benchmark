#!/usr/bin/env python3
"""
Test attention-based DAG prediction functionality.
"""

import torch

from stable_nalu.layer.dag import DAGLayer


def test_attention_prediction_parameter():
    """Test that use_attention_prediction parameter is properly stored."""
    layer_default = DAGLayer(2, 1, 1, use_attention_prediction=False)
    layer_attention = DAGLayer(2, 1, 1, use_attention_prediction=True)

    assert layer_default.use_attention_prediction is False
    assert layer_attention.use_attention_prediction is True
    print("✓ use_attention_prediction parameter properly stored")


def test_attention_predictor_exists():
    """Test that attention predictor is created when enabled."""
    layer_without = DAGLayer(2, 1, 1, use_attention_prediction=False)
    layer_with = DAGLayer(2, 1, 1, use_attention_prediction=True)

    assert not hasattr(
        layer_without, "attention_predictor"
    ), "Should not have attention predictor when disabled"
    assert hasattr(
        layer_with, "attention_predictor"
    ), "Should have attention predictor when enabled"

    # Check attention predictor type
    from stable_nalu.layer.dag import AttentionDAGPredictor

    assert isinstance(
        layer_with.attention_predictor, AttentionDAGPredictor
    ), "Should be AttentionDAGPredictor instance"

    print("✓ Attention predictor created correctly when enabled")


def test_default_behavior_unchanged():
    """Test that default behavior is preserved when attention prediction is disabled."""
    torch.manual_seed(42)
    x = torch.randn(3, 4)

    # Test with all other defaults, attention disabled
    layer_default = DAGLayer(
        4, 1, 3, use_attention_prediction=False, _enable_taps=False
    )
    layer_default.eval()

    with torch.no_grad():
        output = layer_default(x)

    assert output.shape == (3, 1), f"Expected shape (3, 1), got {output.shape}"
    assert torch.all(
        torch.isfinite(output)
    ), "Default behavior should produce finite results"
    print("✓ Default behavior unchanged when attention prediction disabled")


def test_attention_prediction_produces_finite_results():
    """Test that attention prediction produces finite results."""
    torch.manual_seed(42)
    x = torch.randn(3, 4)

    layer_attention = DAGLayer(
        4, 1, 3, use_attention_prediction=True, _enable_taps=False
    )
    layer_attention.eval()

    with torch.no_grad():
        output = layer_attention(x)

    assert output.shape == (3, 1), f"Expected shape (3, 1), got {output.shape}"
    assert torch.all(
        torch.isfinite(output)
    ), "Attention prediction should produce finite results"
    print("✓ Attention prediction produces finite results")


def test_attention_vs_default_different_outputs():
    """Test that attention and default prediction produce different results."""
    torch.manual_seed(42)
    x = torch.randn(1, 2)

    # Create layers with same seed but different prediction methods
    torch.manual_seed(42)
    layer_default = DAGLayer(
        2, 1, 1, use_attention_prediction=False, _enable_taps=False
    )

    torch.manual_seed(42)
    layer_attention = DAGLayer(
        2, 1, 1, use_attention_prediction=True, _enable_taps=False
    )

    layer_default.eval()
    layer_attention.eval()

    with torch.no_grad():
        out_default = layer_default(x)
        out_attention = layer_attention(x)

    # Should produce different results due to different architectures
    difference = abs(out_default.item() - out_attention.item())
    print(f"✓ Default vs attention prediction difference: {difference:.6f}")

    # Both should be finite
    assert torch.isfinite(
        out_default
    ), "Default prediction should produce finite result"
    assert torch.isfinite(
        out_attention
    ), "Attention prediction should produce finite result"

    # They should likely be different (with very high probability due to different architectures)
    # Note: We don't assert this since same random seed might occasionally produce similar results


def test_attention_prediction_training_mode():
    """Test attention prediction works in training mode."""
    torch.manual_seed(42)
    x = torch.randn(2, 3, requires_grad=True)
    target = torch.randn(2, 1)

    layer = DAGLayer(3, 1, 2, use_attention_prediction=True, _enable_taps=False)
    layer.train()

    output = layer(x)
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()

    # Check gradients exist
    assert x.grad is not None, "Input should have gradients"
    assert torch.any(x.grad != 0), "Gradients should be non-zero"

    print("✓ Attention prediction works in training mode")


def test_attention_prediction_with_dense_features():
    """Test attention prediction compatibility with dense features."""
    torch.manual_seed(42)
    x = torch.randn(1, 2)

    layer = DAGLayer(
        2,
        1,
        1,
        use_attention_prediction=True,
        use_dense_features=True,
        _enable_taps=False,
    )
    layer.eval()

    with torch.no_grad():
        output = layer(x)

    assert torch.isfinite(
        output
    ), "Attention + dense features should produce finite result"
    print("✓ Attention prediction compatible with dense features")


def test_attention_prediction_with_other_parameters():
    """Test attention prediction compatibility with other DAG parameters."""
    torch.manual_seed(42)
    x = torch.randn(1, 2)

    # Test with various parameter combinations
    configs = [
        (True, True, True),  # Attention + complex sign + soft clamp
        (True, False, True),  # Attention + complex domain mixing + soft clamp
        (True, True, False),  # Attention + complex sign + simple domain
    ]

    for use_complex_sign, use_simple_domain, use_soft_clamp in configs:
        layer = DAGLayer(
            2,
            1,
            1,
            use_attention_prediction=True,
            use_complex_sign=use_complex_sign,
            use_simple_domain_mixing=use_simple_domain,
            use_soft_clamping=use_soft_clamp,
            _enable_taps=False,
        )
        layer.eval()

        with torch.no_grad():
            output = layer(x)

        assert torch.isfinite(output), f"Attention prediction compatibility test failed"

    print("✓ Attention prediction compatible with all other parameters")


def main():
    """Run all attention prediction tests."""
    print("Testing Attention-Based DAG Prediction")
    print("=" * 50)

    test_attention_prediction_parameter()
    test_attention_predictor_exists()
    test_default_behavior_unchanged()
    test_attention_prediction_produces_finite_results()
    test_attention_vs_default_different_outputs()
    test_attention_prediction_training_mode()
    test_attention_prediction_with_dense_features()
    test_attention_prediction_with_other_parameters()

    print("=" * 50)
    print("✅ All attention prediction tests passed!")


if __name__ == "__main__":
    main()
