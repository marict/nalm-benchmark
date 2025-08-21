#!/usr/bin/env python3
"""
Test DAG layer correctness in training mode vs evaluation mode.
Compares behavior when output selector logits are set to extremely correct values.
"""

import torch

from stable_nalu.layer.dag import DAGLayer


def test_training_vs_eval_correctness():
    """Test arithmetic correctness in both training and eval modes with perfect logits."""
    print("=== Testing Training vs Eval Mode Correctness ===")
    
    # Test cases: (op_name, a, b, expected, O_sign_a, O_sign_b, G_domain, domain_name)
    test_cases = [
        # Linear domain operations
        ("add_pos", 2.0, 3.0, 5.0, 1.0, 1.0, 1.0, "Linear"),
        ("sub_pos", 5.0, 2.0, 3.0, 1.0, -1.0, 1.0, "Linear"),
        ("add_neg", -2.0, -3.0, -5.0, 1.0, 1.0, 1.0, "Linear"),
        
        # Log domain operations  
        ("mul_pos", 2.0, 3.0, 6.0, 1.0, 1.0, 0.0, "Log"),
        ("div_pos", 6.0, 2.0, 3.0, 1.0, -1.0, 0.0, "Log"),
        ("mul_neg", -2.0, -3.0, 6.0, 1.0, 1.0, 0.0, "Log"),
    ]
    
    results_training = []
    results_eval = []
    
    for op_name, a, b, expected, sign_a, sign_b, G, domain_name in test_cases:
        print(f"\n--- {op_name}: {a} op {b} = {expected} ({domain_name}) ---")
        
        # Test in training mode
        training_result, training_error = test_single_operation_mode(
            op_name, a, b, expected, sign_a, sign_b, G, domain_name, training_mode=True
        )
        results_training.append((op_name, training_result, training_error))
        
        # Test in eval mode
        eval_result, eval_error = test_single_operation_mode(
            op_name, a, b, expected, sign_a, sign_b, G, domain_name, training_mode=False
        )
        results_eval.append((op_name, eval_result, eval_error))
        
        # Compare discrepancy
        discrepancy = abs(training_result - eval_result)
        print(f"  Training result: {training_result:8.6f}")
        print(f"  Eval result:     {eval_result:8.6f}")
        print(f"  Discrepancy:     {discrepancy:8.6f}")
        print(f"  Training error:  {training_error:8.6f}")
        print(f"  Eval error:      {eval_error:8.6f}")
        
        if discrepancy > 1e-6:
            print(f"  ⚠️  SIGNIFICANT DISCREPANCY!")
        else:
            print(f"  ✅ Results match closely")
    
    # Summary
    print(f"\n=== Summary ===")
    training_successes = sum(1 for _, result, error in results_training if error < 0.001)
    eval_successes = sum(1 for _, result, error in results_eval if error < 0.001)
    
    print(f"Training mode: {training_successes}/{len(results_training)} operations correct")
    print(f"Eval mode:     {eval_successes}/{len(results_eval)} operations correct")
    
    # Check for significant discrepancies
    max_discrepancy = 0.0
    for i, (op_name, _, _) in enumerate(results_training):
        training_result = results_training[i][1]
        eval_result = results_eval[i][1]
        discrepancy = abs(training_result - eval_result)
        max_discrepancy = max(max_discrepancy, discrepancy)
    
    print(f"Maximum discrepancy: {max_discrepancy:.8f}")
    
    if max_discrepancy < 1e-6:
        print("✅ Training and eval modes produce virtually identical results")
    else:
        print("⚠️  Training and eval modes have measurable differences")
    
    return max_discrepancy < 1e-6


def test_single_operation_mode(
    op_name, input_a, input_b, expected, manual_O_sign_a, manual_O_sign_b, 
    manual_G, domain_name, training_mode=True
):
    """Test a specific operation in either training or eval mode."""
    
    # Create layer with manual weights
    layer = DAGLayer(4, 1, 3, enable_taps=False, _do_not_predict_weights=True)
    
    if training_mode:
        layer.train()
    else:
        layer.eval()
    
    device = next(layer.parameters()).device
    dtype = torch.float32
    
    # Set up manual weights
    layer.test_O_mag = torch.zeros(1, 3, layer.total_nodes, dtype=dtype, device=device)
    layer.test_O_sign = torch.zeros(1, 3, layer.total_nodes, dtype=dtype, device=device)
    layer.test_G = torch.zeros(1, 3, dtype=dtype, device=device)
    layer.test_out_logits = torch.zeros(1, 3, dtype=dtype, device=device)
    
    # Set operand selectors for step 0
    layer.test_O_mag[0, 0, 0] = 1.0  # Select input[0]
    layer.test_O_mag[0, 0, 1] = 1.0  # Select input[1]
    layer.test_O_sign[0, 0, 0] = manual_O_sign_a  # Sign for input[0]
    layer.test_O_sign[0, 0, 1] = manual_O_sign_b  # Sign for input[1]
    
    # Set domain
    layer.test_G[0, 0] = manual_G
    
    # Output selector - use EXTREMELY strong logits to ensure clear selection
    layer.test_out_logits[0, 0] = 1000.0  # Select first computed node (step 0)
    layer.test_out_logits[0, 1] = -1000.0  # Strongly avoid step 1
    layer.test_out_logits[0, 2] = -1000.0  # Strongly avoid step 2
    
    # Create test input and run
    test_input = torch.tensor([[input_a, input_b, 0.0, 0.0]], dtype=dtype)
    
    with torch.no_grad():
        output = layer(test_input)
    
    result = output.item()
    error = abs(result - expected)
    
    return result, error


def test_extreme_logit_values():
    """Test with various extreme logit values to see effect on training/eval discrepancy."""
    print("\n=== Testing Various Extreme Logit Values ===")
    
    # Simple addition test case
    op_name, a, b, expected = "add_test", 2.0, 3.0, 5.0
    sign_a, sign_b, G, domain = 1.0, 1.0, 1.0, "Linear"
    
    logit_strengths = [1.0, 10.0, 100.0, 1000.0, 10000.0]
    
    for strength in logit_strengths:
        print(f"\n--- Logit strength: ±{strength} ---")
        
        # Test both modes with this logit strength
        training_result, _ = test_single_operation_with_logit_strength(
            a, b, expected, sign_a, sign_b, G, strength, training_mode=True
        )
        eval_result, _ = test_single_operation_with_logit_strength(
            a, b, expected, sign_a, sign_b, G, strength, training_mode=False
        )
        
        discrepancy = abs(training_result - eval_result)
        print(f"  Training: {training_result:10.8f}")
        print(f"  Eval:     {eval_result:10.8f}")
        print(f"  Discrepancy: {discrepancy:10.8f}")
        
        if discrepancy < 1e-10:
            print(f"  ✅ Virtually identical")
        elif discrepancy < 1e-6:
            print(f"  ✅ Very close")
        else:
            print(f"  ⚠️  Measurable difference")


def test_single_operation_with_logit_strength(
    input_a, input_b, expected, manual_O_sign_a, manual_O_sign_b, 
    manual_G, logit_strength, training_mode=True
):
    """Test operation with specified logit strength."""
    
    layer = DAGLayer(4, 1, 3, enable_taps=False, _do_not_predict_weights=True)
    
    if training_mode:
        layer.train()
    else:
        layer.eval()
    
    device = next(layer.parameters()).device
    dtype = torch.float32
    
    # Set up manual weights
    layer.test_O_mag = torch.zeros(1, 3, layer.total_nodes, dtype=dtype, device=device)
    layer.test_O_sign = torch.zeros(1, 3, layer.total_nodes, dtype=dtype, device=device)
    layer.test_G = torch.zeros(1, 3, dtype=dtype, device=device)
    layer.test_out_logits = torch.zeros(1, 3, dtype=dtype, device=device)
    
    # Set operand selectors
    layer.test_O_mag[0, 0, 0] = 1.0
    layer.test_O_mag[0, 0, 1] = 1.0
    layer.test_O_sign[0, 0, 0] = manual_O_sign_a
    layer.test_O_sign[0, 0, 1] = manual_O_sign_b
    layer.test_G[0, 0] = manual_G
    
    # Set logits with specified strength
    layer.test_out_logits[0, 0] = logit_strength
    layer.test_out_logits[0, 1] = -logit_strength
    layer.test_out_logits[0, 2] = -logit_strength
    
    test_input = torch.tensor([[input_a, input_b, 0.0, 0.0]], dtype=dtype)
    
    with torch.no_grad():
        output = layer(test_input)
    
    result = output.item()
    error = abs(result - expected)
    
    return result, error


def main():
    """Run all training mode correctness tests."""
    print("DAG Layer Training Mode Correctness Tests")
    print("=" * 50)
    
    # Test 1: Basic training vs eval comparison
    basic_correctness = test_training_vs_eval_correctness()
    
    # Test 2: Effect of extreme logit values
    test_extreme_logit_values()
    
    print(f"\n" + "=" * 50)
    print("FINAL SUMMARY")
    print("=" * 50)
    
    if basic_correctness:
        print("✅ Training and eval modes produce consistent results")
    else:
        print("⚠️  Training and eval modes show discrepancies")
    
    return basic_correctness


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)