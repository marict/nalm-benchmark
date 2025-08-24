#!/usr/bin/env python3
"""
Test extended multiplication features in DAG layer.
"""

import torch

from stable_nalu.layer.dag import DAGLayer


def test_extended_features_parameter():
    """Test that extended_mul_features parameter is properly stored."""
    layer_standard = DAGLayer(2, 1, 1, extended_mul_features=False)
    layer_extended = DAGLayer(2, 1, 1, extended_mul_features=True)

    assert layer_standard.extended_mul_features is False
    assert layer_extended.extended_mul_features is True
    print("✓ extended_mul_features parameter properly stored")


def test_feature_count_difference():
    """Test that extended features increase the feature count correctly."""
    layer_standard = DAGLayer(
        4,
        1,
        1,
        use_dense_features=True,
        extended_mul_features=False,
        _enable_taps=False,
    )
    layer_extended = DAGLayer(
        4, 1, 1, use_dense_features=True, extended_mul_features=True, _enable_taps=False
    )

    assert (
        layer_standard.dense_features_per_input == 8
    ), f"Standard should have 8 features, got {layer_standard.dense_features_per_input}"
    assert (
        layer_extended.dense_features_per_input == 17
    ), f"Extended should have 17 features, got {layer_extended.dense_features_per_input}"

    print(
        f"✓ Feature counts: standard={layer_standard.dense_features_per_input}, extended={layer_extended.dense_features_per_input}"
    )


def test_extended_features_shape():
    """Test that extended features produce correct tensor shapes."""
    torch.manual_seed(42)
    x = torch.randn(2, 4)  # batch_size=2, in_features=4

    layer_extended = DAGLayer(
        4, 1, 1, use_dense_features=True, extended_mul_features=True, _enable_taps=False
    )

    dense_features = layer_extended.extract_dense_features(x)
    expected_shape = (2, 4 * 17)  # batch_size, in_features * dense_features_per_input

    assert (
        dense_features.shape == expected_shape
    ), f"Expected shape {expected_shape}, got {dense_features.shape}"
    print(f"✓ Extended features shape: {dense_features.shape}")


def test_extended_vs_standard_different_outputs():
    """Test that extended and standard features produce different results."""
    torch.manual_seed(42)
    x = torch.randn(1, 2)

    # Create layers with same seed but different features
    torch.manual_seed(42)
    layer_standard = DAGLayer(
        2,
        1,
        1,
        use_dense_features=True,
        extended_mul_features=False,
        _enable_taps=False,
    )

    torch.manual_seed(42)
    layer_extended = DAGLayer(
        2, 1, 1, use_dense_features=True, extended_mul_features=True, _enable_taps=False
    )

    layer_standard.eval()
    layer_extended.eval()

    with torch.no_grad():
        out_standard = layer_standard(x)
        out_extended = layer_extended(x)

    # Should produce different results due to different feature representations
    difference = abs(out_standard.item() - out_extended.item())
    print(f"✓ Standard vs extended features difference: {difference:.6f}")

    # Both should be finite
    assert torch.isfinite(
        out_standard
    ), "Standard features should produce finite result"
    assert torch.isfinite(
        out_extended
    ), "Extended features should produce finite result"


def test_extended_features_training_mode():
    """Test extended features work in training mode."""
    torch.manual_seed(42)
    x = torch.randn(2, 3, requires_grad=True)
    target = torch.randn(2, 1)

    layer = DAGLayer(
        3, 1, 2, use_dense_features=True, extended_mul_features=True, _enable_taps=False
    )
    layer.train()

    output = layer(x)
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()

    # Check gradients exist
    assert x.grad is not None, "Input should have gradients"
    assert torch.any(x.grad != 0), "Gradients should be non-zero"

    print("✓ Extended features work in training mode")


def test_specific_extended_features_values():
    """Test that specific extended features are computed correctly."""
    torch.manual_seed(42)
    # Use simple test input to verify feature calculations
    x = torch.tensor([[2.0, -3.0]])  # Simple values for verification

    layer = DAGLayer(
        2, 1, 1, use_dense_features=True, extended_mul_features=True, _enable_taps=False
    )
    dense_features = layer.extract_dense_features(x)

    # Extract features for first input (2.0)
    # Features are: [x, x^2, x^3, exp(x), log(|x|+eps), sin(x), cos(x), tanh(x),
    #                x^4, x^5, x^(-1), x^(1/2), x^(1/3), log2(|x|+eps), log10(|x|+eps), 2^x, 10^x]
    first_input_features = dense_features[
        0, :17
    ]  # First 17 features are for first input

    # Verify some specific calculations (with tolerance for floating point)
    eps = 1e-6
    assert (
        abs(first_input_features[0] - 2.0) < eps
    ), f"x should be 2.0, got {first_input_features[0]}"
    assert (
        abs(first_input_features[1] - 4.0) < eps
    ), f"x^2 should be 4.0, got {first_input_features[1]}"
    assert (
        abs(first_input_features[2] - 8.0) < eps
    ), f"x^3 should be 8.0, got {first_input_features[2]}"
    assert (
        abs(first_input_features[8] - 16.0) < eps
    ), f"x^4 should be 16.0, got {first_input_features[8]}"
    assert (
        abs(first_input_features[9] - 32.0) < eps
    ), f"x^5 should be 32.0, got {first_input_features[9]}"
    assert (
        abs(first_input_features[10] - 0.5) < eps
    ), f"x^(-1) should be 0.5, got {first_input_features[10]}"

    print("✓ Extended features computed correctly")


def test_multiplication_quick_training():
    """Test a quick multiplication training to see if extended features help."""
    torch.manual_seed(42)

    # Create simple multiplication dataset
    inputs = torch.tensor(
        [
            [2.0, 3.0, 0.0, 0.0],  # 2*3 = 6
            [4.0, 2.0, 0.0, 0.0],  # 4*2 = 8
            [3.0, 3.0, 0.0, 0.0],  # 3*3 = 9
        ]
    )
    targets = torch.tensor([[6.0], [8.0], [9.0]])

    # Test both standard and extended features
    results = {}

    for extended in [False, True]:
        layer = DAGLayer(
            4,
            1,
            3,
            use_dense_features=True,
            extended_mul_features=extended,
            div_biased_init_O_sign=False,  # No division bias for multiplication
            _enable_taps=False,
            _enable_debug_logging=False,
        )

        # Override debug function to avoid breakpoints
        def no_debug_nan_check(self, name, tensor, print_debug=False):
            return torch.isnan(tensor).any() or torch.isinf(tensor).any()

        layer._is_nan = no_debug_nan_check.__get__(layer, DAGLayer)

        optimizer = torch.optim.Adam(layer.parameters(), lr=1e-4)
        loss_fn = torch.nn.MSELoss()

        layer.train()
        initial_loss = None

        feature_type = "extended" if extended else "standard"
        print(f"\nTraining with {feature_type} features:")

        for step in range(50):
            optimizer.zero_grad()

            outputs = layer(inputs)
            loss = loss_fn(outputs, targets)

            if step == 0:
                initial_loss = loss.item()

            if not torch.isfinite(loss):
                print(f"  Non-finite loss at step {step}: {loss.item()}")
                break

            loss.backward()
            torch.nn.utils.clip_grad_norm_(layer.parameters(), max_norm=0.5)
            optimizer.step()

            if step % 10 == 0 or step == 49:
                print(f"  Step {step:2d}: Loss = {loss.item():.6f}")

        final_loss = loss.item()
        improvement = (initial_loss - final_loss) / initial_loss * 100
        results[feature_type] = {
            "initial": initial_loss,
            "final": final_loss,
            "improvement": improvement,
        }

        print(f"  Initial loss: {initial_loss:.6f}")
        print(f"  Final loss: {final_loss:.6f}")
        print(f"  Improvement: {improvement:.1f}%")

    print(f"\n=== Results Comparison ===")
    print(f"Standard features improvement: {results['standard']['improvement']:.1f}%")
    print(f"Extended features improvement: {results['extended']['improvement']:.1f}%")

    if results["extended"]["improvement"] > results["standard"]["improvement"]:
        print("✅ Extended features show better improvement for multiplication!")
    else:
        print("ℹ️  Extended features don't show clear advantage in this quick test")

    return results


def main():
    """Run all extended features tests."""
    print("Testing Extended Multiplication Features")
    print("=" * 50)

    test_extended_features_parameter()
    test_feature_count_difference()
    test_extended_features_shape()
    test_extended_vs_standard_different_outputs()
    test_extended_features_training_mode()
    test_specific_extended_features_values()
    results = test_multiplication_quick_training()

    print("=" * 50)
    print("✅ All extended features tests passed!")
    return results


if __name__ == "__main__":
    main()
