from __future__ import annotations

import torch

from stable_nalu.layer.dag import DAGLayer


def _basic_forward(layer: DAGLayer) -> None:
    layer.eval()
    x = torch.tensor([[1.0, -2.0], [0.5, 0.25]], dtype=torch.float32)
    y = layer(x)
    assert y.shape == (x.shape[0], 1)
    assert torch.isfinite(y).all(), "Output must be finite"


def test_dag_basic_linear_selector() -> None:
    layer = DAGLayer(
        in_features=2,
        out_features=1,
        dag_depth=1,
        use_attention_selector=False,
        _enable_taps=False,
    )
    _basic_forward(layer)


def test_dag_attention_selector() -> None:
    layer = DAGLayer(
        in_features=2,
        out_features=1,
        dag_depth=1,
        use_attention_selector=True,
        selector_dim=8,
        _enable_taps=False,
    )
    _basic_forward(layer)


def test_dag_attention_with_positional_encoding() -> None:
    # Multiple steps to exercise step embeddings
    layer = DAGLayer(
        in_features=3,
        out_features=1,
        dag_depth=2,
        use_attention_selector=True,
        selector_dim=8,
        use_positional_encoding=True,
        _enable_taps=False,
    )
    layer.eval()
    x = torch.randn(4, 3)
    y = layer(x)
    assert y.shape == (4, 1)
    assert torch.isfinite(y).all()


def test_dag_gradient_flow() -> None:
    """Test that gradients can flow through the DAG layer during backpropagation."""
    layer = DAGLayer(
        in_features=2,
        out_features=1,
        dag_depth=1,
        use_attention_selector=False,
        _enable_taps=False,
    )
    layer.train()

    # Create input that requires gradients
    x = torch.tensor(
        [[1.0, -2.0], [0.5, 0.25]], dtype=torch.float32, requires_grad=True
    )

    # Forward pass
    y = layer(x)

    # Create a simple loss (sum of outputs)
    loss = y.sum()

    # Backward pass
    loss.backward()

    # Check that input gradients exist and are finite
    assert x.grad is not None, "Input gradients should exist"
    assert torch.isfinite(x.grad).all(), "Input gradients must be finite"
    assert (x.grad != 0).any(), "Some input gradients should be non-zero"

    # Check that layer parameter gradients exist and are finite
    has_param_grads = False
    for name, param in layer.named_parameters():
        if param.grad is not None:
            has_param_grads = True
            assert torch.isfinite(
                param.grad
            ).all(), f"Parameter {name} gradients must be finite"

    assert has_param_grads, "At least some parameters should have gradients"


def test_dag_gradient_flow_attention() -> None:
    """Test gradient flow through DAG layer with attention selector."""
    layer = DAGLayer(
        in_features=3,
        out_features=1,
        dag_depth=2,
        use_attention_selector=True,
        selector_dim=8,
        _enable_taps=False,
    )
    layer.train()

    # Create input with gradients
    x = torch.randn(2, 3, requires_grad=True)

    # Forward and backward
    y = layer(x)
    loss = y.sum()
    loss.backward()

    # Verify input gradients
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert (x.grad != 0).any()

    # Verify parameter gradients
    for name, param in layer.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(
                param.grad
            ).all(), f"Parameter {name} has non-finite gradients"


def test_dag_gradient_magnitude_stability() -> None:
    """Test that gradients remain stable across different input magnitudes."""
    layer = DAGLayer(
        in_features=2,
        out_features=1,
        dag_depth=1,
        use_attention_selector=False,
        _enable_taps=False,
    )
    layer.train()

    # Test with different input scales
    scales = [1e-3, 1.0, 1e3]

    for scale in scales:
        x = torch.tensor([[1.0, -2.0]], dtype=torch.float32, requires_grad=True) * scale

        y = layer(x)
        loss = y.sum()
        loss.backward()

        # Check gradients are finite regardless of input scale
        assert x.grad is not None
        assert torch.isfinite(
            x.grad
        ).all(), f"Input gradients not finite for scale {scale}"

        # Clear gradients for next iteration
        layer.zero_grad()


def test_dag_gradient_magnitudes() -> None:
    """Test that gradients are non-zero and within expected ranges."""
    layer = DAGLayer(
        in_features=2,
        out_features=1,
        dag_depth=1,
        use_attention_selector=False,
        _enable_taps=False,
    )
    layer.train()

    # Create input that requires gradients
    x = torch.tensor(
        [[1.0, -2.0], [0.5, 0.25]], dtype=torch.float32, requires_grad=True
    )

    # Forward pass
    y = layer(x)

    # Create a simple loss (sum of outputs)
    loss = y.sum()

    # Backward pass
    loss.backward()

    # Check input gradient magnitudes
    assert x.grad is not None
    grad_magnitudes = x.grad.abs()
    assert (
        grad_magnitudes > 1e-6
    ).any(), "Some input gradients should be reasonably large"
    assert (
        grad_magnitudes < 1e2
    ).all(), "Input gradients should not be excessively large"

    # Check parameter gradient magnitudes
    for name, param in layer.named_parameters():
        if param.grad is not None:
            param_grad_mag = param.grad.abs()
            # Skip zero gradients (some parameters may not affect the loss)
            if param_grad_mag.max() > 0:
                assert (
                    param_grad_mag.max() < 1e2
                ), f"Parameter {name} gradients too large: {param_grad_mag.max()}"
                assert (
                    param_grad_mag.max() > 1e-8
                ), f"Parameter {name} gradients too small: {param_grad_mag.max()}"


def test_dag_gradient_flow_comprehensive() -> None:
    """Comprehensive test of gradient flow including magnitude checks."""
    layer = DAGLayer(
        in_features=3,
        out_features=1,
        dag_depth=2,
        use_attention_selector=True,
        selector_dim=8,
        _enable_taps=False,
    )
    layer.train()

    # Create input with gradients
    x = torch.randn(4, 3, requires_grad=True)

    # Forward and backward
    y = layer(x)
    loss = y.mean()  # Use mean to normalize by batch size
    loss.backward()

    # Verify input gradients exist and are reasonable
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert (x.grad != 0).any()

    # Check input gradient magnitude range
    input_grad_mag = x.grad.abs()
    assert input_grad_mag.max() > 1e-6, "Input gradients should be significant"
    assert input_grad_mag.max() < 1e2, "Input gradients should not explode"

    # Verify parameter gradients with detailed checks
    params_with_grads = 0
    for name, param in layer.named_parameters():
        if param.grad is not None:
            params_with_grads += 1
            assert torch.isfinite(
                param.grad
            ).all(), f"Parameter {name} has non-finite gradients"

            # Check for reasonable gradient magnitudes (skip if all zero)
            param_grad_mag = param.grad.abs()
            if param_grad_mag.max() > 0:
                assert (
                    param_grad_mag.max() < 1e3
                ), f"Parameter {name} gradients too large"

    assert params_with_grads > 0, "At least some parameters should have gradients"
