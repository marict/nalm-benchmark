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
        in_features=2, out_features=1, dag_depth=1, use_attention_selector=False
    )
    _basic_forward(layer)


def test_dag_attention_selector() -> None:
    layer = DAGLayer(
        in_features=2,
        out_features=1,
        dag_depth=1,
        use_attention_selector=True,
        selector_dim=8,
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
    )
    layer.eval()
    x = torch.randn(4, 3)
    y = layer(x)
    assert y.shape == (4, 1)
    assert torch.isfinite(y).all()
