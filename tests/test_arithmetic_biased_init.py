#!/usr/bin/env python3
"""
Test arithmetic-biased initialization functionality in DAG layer.
"""

import torch

from stable_nalu.layer.dag import DAGLayer


def test_div_biased_init_parameters():
    """Test that div_biased_init parameters are properly stored."""
    layer_default = DAGLayer(
        2,
        1,
        1,
        div_biased_init_G=False,
        div_biased_init_O_select=False,
        div_biased_init_O_sign=False,
    )
    layer_biased = DAGLayer(
        2,
        1,
        1,
        div_biased_init_G=True,
        div_biased_init_O_select=True,
        div_biased_init_O_sign=True,
    )

    assert layer_default.div_biased_init_G is False
    assert layer_default.div_biased_init_O_select is False
    assert layer_default.div_biased_init_O_sign is False

    assert layer_biased.div_biased_init_G is True
    assert layer_biased.div_biased_init_O_select is True
    assert layer_biased.div_biased_init_O_sign is True
    print("✓ div_biased_init parameters properly stored")


def test_default_initialization_unchanged():
    """Test that default initialization behavior is preserved when flags are False."""
    torch.manual_seed(42)

    # Test with flags off (should be identical to original behavior)
    layer1 = DAGLayer(
        2,
        1,
        1,
        div_biased_init_G=False,
        div_biased_init_O_select=False,
        div_biased_init_O_sign=False,
        _enable_taps=False,
    )
    layer2 = DAGLayer(
        2,
        1,
        1,
        div_biased_init_G=False,
        div_biased_init_O_select=False,
        div_biased_init_O_sign=False,
        _enable_taps=False,
    )

    # Both should have same initialization pattern (normal weights, zero biases)
    assert torch.allclose(
        layer1.O_mag_head.bias, torch.zeros_like(layer1.O_mag_head.bias)
    )
    assert torch.allclose(
        layer1.O_sign_head.bias, torch.zeros_like(layer1.O_sign_head.bias)
    )
    assert torch.allclose(layer1.G_head.bias, torch.zeros_like(layer1.G_head.bias))
    assert torch.allclose(
        layer1.output_selector_head.bias,
        torch.zeros_like(layer1.output_selector_head.bias),
    )

    print("✓ Default initialization unchanged when flags are False")


def test_biased_initialization_applied():
    """Test that biased initialization is actually applied when flags are True."""
    torch.manual_seed(42)

    layer = DAGLayer(
        4,
        1,
        3,
        div_biased_init_G=True,
        div_biased_init_O_select=True,
        div_biased_init_O_sign=True,
        _enable_taps=False,
    )

    # Check that biases are no longer zero
    assert not torch.allclose(
        layer.O_mag_head.bias, torch.zeros_like(layer.O_mag_head.bias)
    )
    assert not torch.allclose(
        layer.O_sign_head.bias, torch.zeros_like(layer.O_sign_head.bias)
    )
    assert not torch.allclose(layer.G_head.bias, torch.zeros_like(layer.G_head.bias))
    assert not torch.allclose(
        layer.output_selector_head.bias,
        torch.zeros_like(layer.output_selector_head.bias),
    )

    # Check specific bias patterns
    # O_mag_head: input nodes should be positive, computed nodes negative
    total_nodes = layer.total_nodes
    for step in range(layer.dag_depth):
        step_start = step * total_nodes

        # Input nodes (first 4) should be positive
        input_biases = layer.O_mag_head.bias[
            step_start : step_start + layer.num_initial_nodes
        ]
        assert torch.all(
            input_biases > 0
        ), f"Input biases should be positive, got {input_biases}"

        # Computed nodes should be negative
        computed_biases = layer.O_mag_head.bias[
            step_start + layer.num_initial_nodes : step_start + total_nodes
        ]
        assert torch.all(
            computed_biases < 0
        ), f"Computed biases should be negative, got {computed_biases}"

    # O_sign_head: first input positive, second input negative
    assert layer.O_sign_head.bias[0] > 0, "First input sign should be positive"
    assert layer.O_sign_head.bias[1] < 0, "Second input sign should be negative"

    # G_head: biased toward log domain (negative bias)
    assert torch.all(
        layer.G_head.bias < 0
    ), "Domain gate should be biased toward log domain"

    # Output selector: first step preferred
    assert (
        layer.output_selector_head.bias[0] > layer.output_selector_head.bias[1]
    ), "First step should be preferred"

    print("✓ Biased initialization properly applied")


def test_biased_vs_default_different_outputs():
    """Test that biased and default initialization produce different outputs."""
    torch.manual_seed(42)
    x = torch.randn(1, 2)

    # Create layers with same seed but different initialization
    torch.manual_seed(42)
    layer_default = DAGLayer(
        2,
        1,
        1,
        div_biased_init_G=False,
        div_biased_init_O_select=False,
        div_biased_init_O_sign=False,
        _enable_taps=False,
    )

    torch.manual_seed(42)
    layer_biased = DAGLayer(
        2,
        1,
        1,
        div_biased_init_G=True,
        div_biased_init_O_select=True,
        div_biased_init_O_sign=True,
        _enable_taps=False,
    )

    layer_default.eval()
    layer_biased.eval()

    with torch.no_grad():
        out_default = layer_default(x)
        out_biased = layer_biased(x)

    # Should produce different results due to different initialization
    difference = abs(out_default.item() - out_biased.item())
    print(f"✓ Biased vs default initialization difference: {difference:.6f}")

    # Both should be finite
    assert torch.isfinite(
        out_default
    ), "Default initialization should produce finite result"
    assert torch.isfinite(
        out_biased
    ), "Biased initialization should produce finite result"

    # They should actually be different (with high probability)
    assert difference > 1e-6, f"Expected significant difference, got {difference}"


def test_biased_initialization_training_mode():
    """Test biased initialization works in training mode."""
    torch.manual_seed(42)
    x = torch.randn(2, 3, requires_grad=True)
    target = torch.randn(2, 1)

    layer = DAGLayer(
        3,
        1,
        2,
        div_biased_init_G=True,
        div_biased_init_O_select=True,
        div_biased_init_O_sign=True,
        _enable_taps=False,
    )
    layer.train()

    output = layer(x)
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()

    # Check gradients exist
    assert x.grad is not None, "Input should have gradients"
    assert torch.any(x.grad != 0), "Gradients should be non-zero"

    print("✓ Biased initialization works in training mode")


def test_biased_initialization_various_sizes():
    """Test biased initialization with different layer sizes."""
    configs = [
        (2, 1, 1),  # Minimal
        (4, 1, 3),  # Standard
        (6, 1, 5),  # Larger
    ]

    for in_features, out_features, dag_depth in configs:
        layer = DAGLayer(
            in_features,
            out_features,
            dag_depth,
            div_biased_init_G=True,
            div_biased_init_O_select=True,
            div_biased_init_O_sign=True,
            _enable_taps=False,
        )
        layer.eval()

        x = torch.randn(1, in_features)
        with torch.no_grad():
            output = layer(x)

        assert torch.isfinite(
            output
        ), f"Config ({in_features}, {out_features}, {dag_depth}) failed"
        print(
            f"✓ Biased initialization works with config ({in_features}, {out_features}, {dag_depth})"
        )


def test_biased_initialization_with_other_parameters():
    """Test biased initialization compatibility with other parameters."""
    torch.manual_seed(42)
    x = torch.randn(1, 2)

    # Test various parameter combinations with biased init
    configs = [
        (True, False, False),  # Only G bias
        (False, True, False),  # Only O_select bias
        (False, False, True),  # Only O_sign bias
        (True, True, True),  # All biases
    ]

    for G_bias, select_bias, sign_bias in configs:
        layer = DAGLayer(
            2,
            1,
            1,
            div_biased_init_G=G_bias,
            div_biased_init_O_select=select_bias,
            div_biased_init_O_sign=sign_bias,
            use_simple_domain_mixing=True,
            use_complex_sign=True,
            use_soft_clamping=True,
            _enable_taps=False,
        )
        layer.eval()

        with torch.no_grad():
            output = layer(x)

        assert torch.isfinite(output), f"Biased init compatibility test failed"

    print("✓ Biased initialization compatible with all other parameters")


def test_individual_bias_flags():
    """Test that individual bias flags work correctly in isolation."""
    torch.manual_seed(42)

    # Test G bias only
    layer_G = DAGLayer(
        2,
        1,
        1,
        div_biased_init_G=True,
        div_biased_init_O_select=False,
        div_biased_init_O_sign=False,
        _enable_taps=False,
    )
    assert torch.all(layer_G.G_head.bias < 0), "G bias should be negative (log domain)"
    assert torch.allclose(
        layer_G.O_sign_head.bias, torch.zeros_like(layer_G.O_sign_head.bias)
    ), "O_sign should be zero when not biased"
    assert (
        layer_G.output_selector_head.bias[0] == 0
    ), "Output selector should be zero when not biased"

    # Test O_select bias only
    layer_select = DAGLayer(
        2,
        1,
        1,
        div_biased_init_G=False,
        div_biased_init_O_select=True,
        div_biased_init_O_sign=False,
        _enable_taps=False,
    )
    assert torch.allclose(
        layer_select.G_head.bias, torch.zeros_like(layer_select.G_head.bias)
    ), "G should be zero when not biased"
    assert (
        layer_select.output_selector_head.bias[0] > 0
    ), "First output step should be favored"

    # Test O_sign bias only
    layer_sign = DAGLayer(
        2,
        1,
        1,
        div_biased_init_G=False,
        div_biased_init_O_select=False,
        div_biased_init_O_sign=True,
        _enable_taps=False,
    )
    assert torch.allclose(
        layer_sign.G_head.bias, torch.zeros_like(layer_sign.G_head.bias)
    ), "G should be zero when not biased"
    assert (
        layer_sign.O_sign_head.bias[0] > 0 and layer_sign.O_sign_head.bias[1] < 0
    ), "Should have division sign pattern"

    print("✓ Individual bias flags work correctly in isolation")


def main():
    """Run all division-biased initialization tests."""
    print("Testing Division-Biased Initialization")
    print("=" * 50)

    test_div_biased_init_parameters()
    test_default_initialization_unchanged()
    test_biased_initialization_applied()
    test_biased_vs_default_different_outputs()
    test_biased_initialization_training_mode()
    test_biased_initialization_various_sizes()
    test_biased_initialization_with_other_parameters()
    test_individual_bias_flags()

    print("=" * 50)
    print("✅ All division-biased initialization tests passed!")


if __name__ == "__main__":
    main()
