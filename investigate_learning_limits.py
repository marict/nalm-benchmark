#!/usr/bin/env python3
"""
Investigate what might be limiting the DAG model's learning capability.
"""

import numpy as np
import torch

from stable_nalu.layer.dag import DAGLayer


def analyze_gradient_flow():
    """Analyze how gradients flow through the DAG layer."""
    print("=== Gradient Flow Analysis ===")

    layer = DAGLayer(4, 1, 3, enable_taps=False)
    layer.train()

    # Create a simple arithmetic problem: 2 + 3 = 5
    inputs = torch.tensor([[2.0, 3.0, 1.0, 1.0]], requires_grad=True)
    target = torch.tensor([[5.0]])

    # Forward pass
    output = layer(inputs)
    loss = torch.nn.functional.mse_loss(output, target)

    print(
        f"Initial output: {output.item():.6f}, Target: {target.item():.1f}, Loss: {loss.item():.6f}"
    )

    # Backward pass
    loss.backward()

    # Analyze gradients by component
    print("\nGradient magnitudes by component:")
    for name, param in layer.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            param_norm = param.norm().item()
            relative_grad = grad_norm / (param_norm + 1e-8)
            print(
                f"  {name:25s}: grad_norm={grad_norm:.2e}, param_norm={param_norm:.2e}, ratio={relative_grad:.2e}"
            )

    # Check input gradients
    input_grad_norm = inputs.grad.norm().item()
    print(f"  {'input_gradients':25s}: grad_norm={input_grad_norm:.2e}")

    return loss.item()


def analyze_parameter_initialization():
    """Analyze if parameter initialization might be limiting learning."""
    print("\n=== Parameter Initialization Analysis ===")

    layer = DAGLayer(4, 1, 3, enable_taps=False)

    print("Initial parameter statistics:")
    for name, param in layer.named_parameters():
        mean_val = param.mean().item()
        std_val = param.std().item()
        min_val = param.min().item()
        max_val = param.max().item()
        print(
            f"  {name:25s}: mean={mean_val:.4f}, std={std_val:.4f}, range=[{min_val:.4f}, {max_val:.4f}]"
        )


def analyze_gate_learning_dynamics():
    """Analyze how gates evolve during learning."""
    print("\n=== Gate Learning Dynamics ===")

    layer = DAGLayer(4, 1, 3, enable_taps=False)

    # Simple training loop to see how gates evolve
    inputs = torch.tensor(
        [[2.0, 3.0, 1.0, 1.0], [1.0, 4.0, 1.0, 1.0], [3.0, 2.0, 1.0, 1.0]]
    )
    targets = torch.tensor([[5.0], [5.0], [5.0]])

    optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)

    print("Gate evolution over training steps:")
    for step in range(5):
        layer.train()
        optimizer.zero_grad()

        outputs = layer(inputs)
        loss = torch.nn.functional.mse_loss(outputs, targets)
        loss.backward()
        optimizer.step()

        # Check gate values
        layer.eval()
        with torch.no_grad():
            _ = layer(inputs)
            if hasattr(layer, "_last_G"):
                G_values = layer._last_G
                G_mean = G_values.mean(dim=0)  # Average across batch
                print(
                    f"  Step {step}: Loss={loss.item():.4f}, G_means={G_mean.numpy()}"
                )


def analyze_operand_selector_behavior():
    """Analyze how operand selectors behave."""
    print("\n=== Operand Selector Analysis ===")

    layer = DAGLayer(4, 1, 3, enable_taps=False)
    layer.eval()

    # Test with simple input
    inputs = torch.tensor([[2.0, 3.0, 1.0, 1.0]])

    with torch.no_grad():
        output = layer(inputs)

        if hasattr(layer, "_last_O"):
            O_values = layer._last_O[0]  # First batch element
            print(f"Operand selector values (first batch):")
            for step in range(layer.dag_depth):
                step_values = O_values[step]
                print(f"  Step {step}: {step_values.numpy()}")

                # Check if selectors are learning meaningful patterns
                abs_values = step_values.abs()
                max_val = abs_values.max().item()
                mean_val = abs_values.mean().item()
                print(f"    Max: {max_val:.4f}, Mean: {mean_val:.4f}")


def analyze_magnitude_vs_sign_interaction():
    """Analyze how magnitude and sign components interact."""
    print("\n=== Magnitude vs Sign Interaction ===")

    layer = DAGLayer(4, 1, 3, enable_taps=False)
    layer.eval()

    # Test with various sign combinations
    test_cases = [
        torch.tensor([[2.0, 3.0, 1.0, 1.0]]),  # All positive
        torch.tensor([[-2.0, 3.0, 1.0, 1.0]]),  # Mixed signs
        torch.tensor([[-2.0, -3.0, 1.0, 1.0]]),  # All negative
    ]

    for i, inputs in enumerate(test_cases):
        with torch.no_grad():
            output = layer(inputs)

            print(f"Test case {i+1}: {inputs.flatten().numpy()}")
            print(f"  Output: {output.item():.6f}")

            if hasattr(layer, "_last_O_sign") and hasattr(layer, "_last_O_mag"):
                O_sign = layer._last_O_sign[0, 0]  # First batch, first step
                O_mag = layer._last_O_mag[0, 0]
                combined = O_sign * O_mag

                print(f"  First step O_sign: {O_sign.numpy()}")
                print(f"  First step O_mag:  {O_mag.numpy()}")
                print(f"  Combined O:        {combined.numpy()}")


def analyze_computational_bottlenecks():
    """Check for computational bottlenecks that might limit learning."""
    print("\n=== Computational Bottleneck Analysis ===")

    layer = DAGLayer(4, 1, 3, enable_taps=False)

    # Check if certain operations dominate computational load
    inputs = torch.tensor([[2.0, 3.0, 1.0, 1.0]])

    # Manually step through forward pass to check intermediate values
    layer.eval()
    with torch.no_grad():
        output = layer(inputs)

        print("Key computational bounds:")
        print(f"  _mag_min: {layer._mag_min:.2e}")
        print(f"  _mag_max: {layer._mag_max:.2e}")
        print(f"  _log_lim: {layer._log_lim:.4f}")

        # Check if we're hitting these bounds
        if hasattr(layer, "_debug_V_mag_new"):
            for i, mag_values in enumerate(layer._debug_V_mag_new):
                min_mag = mag_values.min().item()
                max_mag = mag_values.max().item()

                hitting_min = min_mag <= layer._mag_min * 10  # Within 10x of minimum
                hitting_max = max_mag >= layer._mag_max * 0.1  # Within 10x of maximum

                print(f"  Step {i} magnitude range: [{min_mag:.2e}, {max_mag:.2e}]")
                if hitting_min:
                    print(f"    ⚠️  Approaching _mag_min bound")
                if hitting_max:
                    print(f"    ⚠️  Approaching _mag_max bound")


def analyze_arithmetic_capability():
    """Test if the layer can represent simple arithmetic operations."""
    print("\n=== Arithmetic Representation Capability ===")

    layer = DAGLayer(4, 1, 3, enable_taps=False)

    # Test systematic arithmetic patterns
    test_operations = [
        ("Addition", [[1.0, 1.0, 0.0, 0.0]], 2.0),
        ("Addition", [[2.0, 3.0, 0.0, 0.0]], 5.0),
        ("Addition", [[5.0, 7.0, 0.0, 0.0]], 12.0),
        ("Subtraction", [[5.0, -2.0, 0.0, 0.0]], 3.0),
        ("Mixed", [[2.0, 3.0, -1.0, 0.0]], 4.0),  # 2+3-1
    ]

    layer.eval()
    results = []

    for op_name, input_vals, expected in test_operations:
        inputs = torch.tensor(input_vals)

        with torch.no_grad():
            output = layer(inputs)
            error = abs(output.item() - expected)
            results.append((op_name, input_vals[0], expected, output.item(), error))

            print(
                f"{op_name:12s}: {input_vals[0]} -> Expected: {expected:5.1f}, Got: {output.item():7.3f}, Error: {error:.3f}"
            )

    # Check if errors show systematic patterns
    errors = [r[4] for r in results]
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    print(f"\nError statistics: Mean={mean_error:.3f}, Std={std_error:.3f}")

    if mean_error > 1.0:
        print("⚠️  High mean error suggests fundamental representation issues")
    if std_error > mean_error:
        print("⚠️  High error variance suggests inconsistent learning")


def main():
    """Run comprehensive learning limitation analysis."""
    print("Investigating DAG Layer Learning Limitations")
    print("=" * 50)

    # Run all analyses
    initial_loss = analyze_gradient_flow()
    analyze_parameter_initialization()
    analyze_gate_learning_dynamics()
    analyze_operand_selector_behavior()
    analyze_magnitude_vs_sign_interaction()
    analyze_computational_bottlenecks()
    analyze_arithmetic_capability()

    print("\n" + "=" * 50)
    print("SUMMARY OF POTENTIAL LEARNING LIMITATIONS:")
    print("=" * 50)

    # Based on initial loss, provide insights
    if initial_loss > 10.0:
        print("🔴 HIGH INITIAL LOSS: Model may have fundamental representation issues")
    elif initial_loss > 1.0:
        print(
            "🟡 MODERATE INITIAL LOSS: Model has learning potential but may be suboptimal"
        )
    else:
        print("🟢 LOW INITIAL LOSS: Model shows good initial representation capability")

    print("\nRecommendations:")
    print("1. Check gradient flow balance across components")
    print("2. Verify gate initialization allows exploration of both domains")
    print("3. Ensure operand selectors can focus on relevant inputs")
    print("4. Monitor if computational bounds are constraining learning")
    print("5. Test with systematic arithmetic patterns to identify representation gaps")


if __name__ == "__main__":
    main()
