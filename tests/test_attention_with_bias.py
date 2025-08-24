#!/usr/bin/env python3
"""
Test attention prediction with biased initialization to ensure they work together correctly.
"""

import torch

from stable_nalu.layer.dag import DAGLayer


def test_attention_with_biased_init():
    """Test that attention prediction works correctly with biased initialization."""
    print("=== Testing Attention + Biased Init Interaction ===")

    torch.manual_seed(42)
    x = torch.tensor([[6.0, 2.0, 0.0, 0.0]])  # Simple division test: 6/2 = 3

    # Test all combinations
    configs = [
        (False, False, "Standard prediction + No bias"),
        (False, True, "Standard prediction + O_sign bias"),
        (True, False, "Attention prediction + No bias"),
        (True, True, "Attention prediction + O_sign bias"),
    ]

    results = []

    for use_attention, use_bias, description in configs:
        print(f"\n--- {description} ---")

        layer = DAGLayer(
            4,
            1,
            3,
            use_attention_prediction=use_attention,
            div_biased_init_O_sign=use_bias,
            _enable_taps=False,
            _enable_debug_logging=False,
        )
        layer.eval()

        with torch.no_grad():
            output = layer(x)

        result = output.item()
        error = abs(result - 3.0)  # Expected result is 3.0 for 6/2

        print(f"  Output: {result:.6f}")
        print(f"  Error from 3.0: {error:.6f}")
        print(f"  Finite: {torch.isfinite(output).item()}")

        results.append((description, result, error, torch.isfinite(output).item()))

        # Check if bias was actually applied when expected
        if use_bias:
            first_step_sign_bias = layer.O_sign_head.bias[0].item()
            second_step_sign_bias = layer.O_sign_head.bias[1].item()
            print(f"  O_sign bias [0]: {first_step_sign_bias:.3f} (should be ~2.0)")
            print(f"  O_sign bias [1]: {second_step_sign_bias:.3f} (should be ~-2.0)")

            # Verify bias is applied correctly
            assert (
                abs(first_step_sign_bias - 2.0) < 0.1
            ), f"Expected first sign bias ~2.0, got {first_step_sign_bias}"
            assert (
                abs(second_step_sign_bias + 2.0) < 0.1
            ), f"Expected second sign bias ~-2.0, got {second_step_sign_bias}"
            print("  ✓ O_sign bias applied correctly")
        else:
            # Check that bias is zero when disabled
            sign_biases = layer.O_sign_head.bias[:4].tolist()  # First 4 biases
            print(f"  O_sign biases: {sign_biases}")
            assert all(
                abs(b) < 0.01 for b in sign_biases
            ), f"Expected zero biases, got {sign_biases}"
            print("  ✓ No bias applied as expected")

    print(f"\n=== Summary ===")
    print(f"{'Configuration':<35} | {'Output':<10} | {'Error':<10} | {'Finite'}")
    print("-" * 70)
    for desc, result, error, finite in results:
        finite_str = "✓" if finite else "❌"
        print(f"{desc:<35} | {result:>8.3f} | {error:>8.3f} | {finite_str:^6}")

    # Check that all results are finite
    all_finite = all(finite for _, _, _, finite in results)
    if all_finite:
        print(f"\n✅ All configurations produce finite results!")
    else:
        print(f"\n❌ Some configurations produced non-finite results!")

    return all_finite


def test_attention_bias_training():
    """Test a quick training run with attention + bias to ensure they work together."""
    print(f"\n=== Testing Training with Attention + Bias ===")

    torch.manual_seed(42)

    # Create simple division dataset
    inputs = torch.tensor(
        [
            [6.0, 2.0, 0.0, 0.0],  # 6/2 = 3
            [8.0, 4.0, 0.0, 0.0],  # 8/4 = 2
            [10.0, 5.0, 0.0, 0.0],  # 10/5 = 2
        ]
    )
    targets = torch.tensor([[3.0], [2.0], [2.0]])

    layer = DAGLayer(
        4,
        1,
        3,
        use_attention_prediction=True,
        div_biased_init_O_sign=True,
        _enable_taps=False,
        _enable_debug_logging=False,
    )

    # Override debug function to avoid breakpoints
    def no_debug_nan_check(self, name, tensor, print_debug=False):
        return torch.isnan(tensor).any() or torch.isinf(tensor).any()

    layer._is_nan = no_debug_nan_check.__get__(layer, DAGLayer)

    optimizer = torch.optim.Adam(layer.parameters(), lr=1e-4)  # Reduced learning rate
    loss_fn = torch.nn.MSELoss()

    layer.train()
    initial_loss = None

    print("Training progress:")
    for step in range(20):
        optimizer.zero_grad()

        outputs = layer(inputs)
        loss = loss_fn(outputs, targets)

        if step == 0:
            initial_loss = loss.item()

        loss.backward()

        # Check for NaN/Inf in loss and gradients
        if not torch.isfinite(loss):
            print(f"  Non-finite loss at step {step}: {loss.item()}")
            break

        # Gradient clipping with more aggressive norm
        torch.nn.utils.clip_grad_norm_(layer.parameters(), max_norm=0.1)

        optimizer.step()

        if step % 5 == 0 or step == 19:
            print(f"  Step {step:2d}: Loss = {loss.item():.6f}")

    final_loss = loss.item()
    improvement = (initial_loss - final_loss) / initial_loss * 100

    print(f"  Initial loss: {initial_loss:.6f}")
    print(f"  Final loss: {final_loss:.6f}")
    print(f"  Improvement: {improvement:.1f}%")

    if improvement > 0:
        print("  ✅ Training improved the loss!")
        return True
    else:
        print("  ❌ Training did not improve the loss")
        return False


def main():
    """Run all tests for attention + bias interaction."""
    print("Testing Attention Prediction with Biased Initialization")
    print("=" * 65)

    test1_success = test_attention_with_biased_init()
    test2_success = test_attention_bias_training()

    print(f"\n" + "=" * 65)
    print("FINAL RESULTS")
    print("=" * 65)

    if test1_success and test2_success:
        print("✅ Attention prediction works correctly with biased initialization!")
        return True
    else:
        print("❌ Issues detected with attention + bias interaction")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
