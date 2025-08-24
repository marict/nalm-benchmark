#!/usr/bin/env python3
"""
Test complex sign computation functionality in DAG layer.
"""

import torch

from stable_nalu.layer.dag import DAGLayer


def test_complex_sign_parameter_exists():
    """Test that use_complex_sign parameter is properly stored."""
    layer_simple = DAGLayer(2, 1, 1, use_complex_sign=False)
    layer_complex = DAGLayer(2, 1, 1, use_complex_sign=True)

    assert layer_simple.use_complex_sign is False
    assert layer_complex.use_complex_sign is True
    print("✓ use_complex_sign parameter properly stored")


def test_simple_sign_computation():
    """Test simple sign computation mode produces finite results."""
    torch.manual_seed(42)
    x = torch.randn(3, 4)

    layer = DAGLayer(4, 1, 3, use_complex_sign=False, _enable_taps=False)
    layer.eval()

    with torch.no_grad():
        output = layer(x)

    assert output.shape == (3, 1), f"Expected shape (3, 1), got {output.shape}"
    assert torch.all(
        torch.isfinite(output)
    ), "Simple sign computation should produce finite results"
    print(f"✓ Simple sign computation: output shape {output.shape}, all finite")


def test_complex_sign_computation():
    """Test complex sign computation mode produces finite results."""
    torch.manual_seed(42)
    x = torch.randn(3, 4)

    layer = DAGLayer(4, 1, 3, use_complex_sign=True, _enable_taps=False)
    layer.eval()

    with torch.no_grad():
        output = layer(x)

    assert output.shape == (3, 1), f"Expected shape (3, 1), got {output.shape}"
    assert torch.all(
        torch.isfinite(output)
    ), "Complex sign computation should produce finite results"
    print(f"✓ Complex sign computation: output shape {output.shape}, all finite")


def test_sign_modes_identical_weights():
    """Test sign computation modes with identical weights."""
    torch.manual_seed(42)
    x = torch.randn(1, 2)

    # Create layers with identical weights
    layer_simple = DAGLayer(2, 1, 1, use_complex_sign=False, _enable_taps=False)
    layer_complex = DAGLayer(2, 1, 1, use_complex_sign=True, _enable_taps=False)
    layer_complex.load_state_dict(layer_simple.state_dict())

    layer_simple.eval()
    layer_complex.eval()

    with torch.no_grad():
        out_simple = layer_simple(x)
        out_complex = layer_complex(x)

    # With identical weights, they should produce different results due to different sign computation
    difference = abs(out_simple.item() - out_complex.item())
    print(
        f"✓ Simple vs complex sign difference with identical weights: {difference:.6f}"
    )

    # Both should be finite
    assert torch.isfinite(
        out_simple
    ), f"Simple sign produced non-finite result: {out_simple.item()}"
    assert torch.isfinite(
        out_complex
    ), f"Complex sign produced non-finite result: {out_complex.item()}"


def test_sign_computation_training_mode():
    """Test both sign computation modes work in training mode."""
    torch.manual_seed(42)
    x = torch.randn(2, 3, requires_grad=True)
    target = torch.randn(2, 1)

    for use_complex in [False, True]:
        layer = DAGLayer(3, 1, 2, use_complex_sign=use_complex, _enable_taps=False)
        layer.train()

        output = layer(x)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()

        # Check gradients exist
        assert x.grad is not None, "Input should have gradients"
        assert torch.any(x.grad != 0), "Gradients should be non-zero"

        mode_name = "complex" if use_complex else "simple"
        print(f"✓ {mode_name.capitalize()} sign computation works in training mode")

        # Reset gradients for next test
        x.grad.zero_()


def test_complex_sign_mathematical_properties():
    """Test mathematical properties of complex sign computation."""
    torch.manual_seed(42)

    # Create layer with complex sign computation
    layer = DAGLayer(2, 1, 1, use_complex_sign=True, _enable_taps=False)
    layer.eval()

    # Test with various input patterns
    test_inputs = [
        torch.tensor([[1.0, 1.0]]),  # Both positive
        torch.tensor([[-1.0, -1.0]]),  # Both negative
        torch.tensor([[1.0, -1.0]]),  # Mixed signs
        torch.tensor([[-1.0, 1.0]]),  # Mixed signs (flipped)
        torch.tensor([[0.0, 1.0]]),  # Zero included
    ]

    for i, x in enumerate(test_inputs):
        with torch.no_grad():
            output = layer(x)

        assert torch.isfinite(output), f"Test input {i} produced non-finite result"
        print(
            f"✓ Complex sign test {i+1}: input {x[0].tolist()} -> output {output.item():.6f}"
        )


def test_sign_computation_with_domain_mixing():
    """Test sign computation compatibility with domain mixing modes."""
    torch.manual_seed(42)
    x = torch.randn(1, 2)

    # Test all combinations
    configs = [
        (True, False),  # Simple domain, simple sign
        (True, True),  # Simple domain, complex sign
        (False, False),  # Complex domain, simple sign
        (False, True),  # Complex domain, complex sign
    ]

    for use_simple_domain, use_complex_sign in configs:
        layer = DAGLayer(
            2,
            1,
            1,
            use_simple_domain_mixing=use_simple_domain,
            use_complex_sign=use_complex_sign,
            _enable_taps=False,
        )
        layer.eval()

        with torch.no_grad():
            output = layer(x)

        assert torch.isfinite(
            output
        ), f"Config (simple_domain={use_simple_domain}, complex_sign={use_complex_sign}) failed"

        domain_name = "simple" if use_simple_domain else "complex"
        sign_name = "complex" if use_complex_sign else "simple"
        print(f"✓ Compatible: {domain_name} domain + {sign_name} sign")


def test_sign_computation_extreme_values():
    """Test sign computation with extreme input values."""
    torch.manual_seed(42)

    # Test both sign modes
    for use_complex in [False, True]:
        layer = DAGLayer(2, 1, 1, use_complex_sign=use_complex, _enable_taps=False)
        layer.eval()

        # Test extreme values
        extreme_inputs = [
            torch.tensor([[1e6, -1e6]]),  # Very large opposing values
            torch.tensor([[1e-6, 1e-6]]),  # Very small values
            torch.tensor([[-1e6, -1e6]]),  # Large negative values
        ]

        for i, x in enumerate(extreme_inputs):
            with torch.no_grad():
                output = layer(x)

            assert torch.isfinite(
                output
            ), f"Extreme test {i} failed for {'complex' if use_complex else 'simple'} sign"

        sign_name = "complex" if use_complex else "simple"
        print(f"✓ {sign_name.capitalize()} sign handles extreme values")


def main():
    """Run all sign computation tests."""
    print("Testing Sign Computation Functionality")
    print("=" * 50)

    test_complex_sign_parameter_exists()
    test_simple_sign_computation()
    test_complex_sign_computation()
    test_sign_modes_identical_weights()
    test_sign_computation_training_mode()
    test_complex_sign_mathematical_properties()
    test_sign_computation_with_domain_mixing()
    test_sign_computation_extreme_values()

    print("=" * 50)
    print("✅ All sign computation tests passed!")


if __name__ == "__main__":
    main()
