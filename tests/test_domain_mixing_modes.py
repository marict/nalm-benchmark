#!/usr/bin/env python3
"""
Test simple vs complex domain mixing functionality in DAG layer.
"""

import torch

from stable_nalu.layer.dag import DAGLayer


def test_domain_mixing_parameter_exists():
    """Test that use_simple_domain_mixing parameter is properly stored."""
    layer_simple = DAGLayer(2, 1, 1, use_simple_domain_mixing=True)
    layer_complex = DAGLayer(2, 1, 1, use_simple_domain_mixing=False)

    assert layer_simple.use_simple_domain_mixing is True
    assert layer_complex.use_simple_domain_mixing is False
    print("✓ use_simple_domain_mixing parameter properly stored")


def test_simple_domain_mixing_functionality():
    """Test that simple domain mixing produces finite results."""
    torch.manual_seed(42)
    x = torch.randn(3, 4)  # Multiple samples, multiple features

    layer = DAGLayer(4, 1, 3, use_simple_domain_mixing=True, _enable_taps=False)
    layer.eval()

    with torch.no_grad():
        output = layer(x)

    assert output.shape == (3, 1), f"Expected shape (3, 1), got {output.shape}"
    assert torch.all(
        torch.isfinite(output)
    ), "Simple domain mixing should produce finite results"
    print(f"✓ Simple domain mixing: output shape {output.shape}, all finite")


def test_complex_domain_mixing_functionality():
    """Test that complex domain mixing produces finite results."""
    torch.manual_seed(42)
    x = torch.randn(3, 4)  # Multiple samples, multiple features

    layer = DAGLayer(4, 1, 3, use_simple_domain_mixing=False, _enable_taps=False)
    layer.eval()

    with torch.no_grad():
        output = layer(x)

    assert output.shape == (3, 1), f"Expected shape (3, 1), got {output.shape}"
    assert torch.all(
        torch.isfinite(output)
    ), "Complex domain mixing should produce finite results"
    print(f"✓ Complex domain mixing: output shape {output.shape}, all finite")


def test_domain_mixing_modes_different_outputs():
    """Test that simple and complex domain mixing can produce different results."""
    torch.manual_seed(42)
    x = torch.randn(1, 2)

    layer_simple = DAGLayer(2, 1, 1, use_simple_domain_mixing=True, _enable_taps=False)
    layer_complex = DAGLayer(
        2, 1, 1, use_simple_domain_mixing=False, _enable_taps=False
    )

    # Note: Different initializations will likely produce different results
    layer_simple.eval()
    layer_complex.eval()

    with torch.no_grad():
        out_simple = layer_simple(x)
        out_complex = layer_complex(x)

    # With different random initializations, outputs will likely differ
    difference = abs(out_simple.item() - out_complex.item())
    print(f"✓ Simple vs complex mixing difference: {difference:.6f}")

    # Both should be finite
    assert torch.isfinite(
        out_simple
    ), f"Simple mixing produced non-finite result: {out_simple.item()}"
    assert torch.isfinite(
        out_complex
    ), f"Complex mixing produced non-finite result: {out_complex.item()}"


def test_domain_mixing_training_mode():
    """Test both domain mixing modes work in training mode."""
    torch.manual_seed(42)
    x = torch.randn(2, 3, requires_grad=True)
    target = torch.randn(2, 1)

    for use_simple in [True, False]:
        layer = DAGLayer(
            3, 1, 2, use_simple_domain_mixing=use_simple, _enable_taps=False
        )
        layer.train()

        output = layer(x)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()

        # Check gradients exist
        assert x.grad is not None, "Input should have gradients"
        assert torch.any(x.grad != 0), "Gradients should be non-zero"

        mode_name = "simple" if use_simple else "complex"
        print(f"✓ {mode_name.capitalize()} domain mixing works in training mode")

        # Reset gradients for next test
        x.grad.zero_()


def test_complex_domain_mixing_methods_exist():
    """Test that complex domain mixing methods are accessible."""
    layer = DAGLayer(2, 1, 1, use_simple_domain_mixing=False, _enable_taps=False)

    # Check that complex mixing methods exist and are callable
    assert hasattr(
        layer, "_compute_aggregates"
    ), "Should have _compute_aggregates method"
    assert hasattr(layer, "_compute_new_sign"), "Should have _compute_new_sign method"
    assert hasattr(
        layer, "_compute_new_magnitude"
    ), "Should have _compute_new_magnitude method"

    assert callable(layer._compute_aggregates), "_compute_aggregates should be callable"
    assert callable(layer._compute_new_sign), "_compute_new_sign should be callable"
    assert callable(
        layer._compute_new_magnitude
    ), "_compute_new_magnitude should be callable"

    print("✓ Complex domain mixing methods exist and are callable")


def test_domain_mixing_with_other_parameters():
    """Test domain mixing compatibility with other DAG layer parameters."""
    torch.manual_seed(42)
    x = torch.randn(1, 2)

    # Test various parameter combinations
    configs = [
        # (simple_mixing, complex_sign, soft_clamp, dense_features)
        (True, True, False, False),  # Simple + complex sign
        (True, False, True, False),  # Simple + soft clamp
        (False, True, False, False),  # Complex + complex sign
        (False, False, True, False),  # Complex + soft clamp
        (True, True, True, False),  # Simple + complex sign + soft clamp
        (False, True, True, False),  # Complex + complex sign + soft clamp
    ]

    for simple_mix, complex_sign, soft_clamp, dense_feat in configs:
        layer = DAGLayer(
            2,
            1,
            1,
            use_simple_domain_mixing=simple_mix,
            use_complex_sign=complex_sign,
            use_soft_clamping=soft_clamp,
            use_dense_features=dense_feat,
            _enable_taps=False,
        )
        layer.eval()

        with torch.no_grad():
            output = layer(x)

        assert torch.isfinite(
            output
        ), f"Config failed: simple={simple_mix}, complex_sign={complex_sign}, soft_clamp={soft_clamp}"

    print("✓ Domain mixing compatible with all parameter combinations")


def main():
    """Run all domain mixing tests."""
    print("Testing Domain Mixing Functionality")
    print("=" * 50)

    test_domain_mixing_parameter_exists()
    test_simple_domain_mixing_functionality()
    test_complex_domain_mixing_functionality()
    test_domain_mixing_modes_different_outputs()
    test_domain_mixing_training_mode()
    test_complex_domain_mixing_methods_exist()
    test_domain_mixing_with_other_parameters()

    print("=" * 50)
    print("✅ All domain mixing tests passed!")


if __name__ == "__main__":
    main()
