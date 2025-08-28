#!/usr/bin/env python3
"""
Test script to verify that sparsity error calculation is working correctly
and producing meaningful values during actual training.
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from stable_nalu.layer.dag import DAGLayer

def test_sparsity_during_simulated_training():
    """Test sparsity calculation with simulated training steps."""
    
    print("=== Testing Sparsity Error During Simulated Training ===")
    
    # Create a DAG layer
    layer = DAGLayer(
        in_features=2,
        out_features=1,
        dag_depth=1,
        _enable_taps=False,
        freeze_O_mul=False  # Allow weights to be learned
    )
    
    # Create simple training data (multiplication task: output = x1 * x2)
    x_train = torch.tensor([
        [2.0, 3.0],
        [1.0, 4.0], 
        [-1.0, 2.0],
        [0.5, 6.0]
    ])
    y_train = torch.tensor([[6.0], [4.0], [-2.0], [3.0]])  # x1 * x2
    
    optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)
    
    print("Initial weights and sparsity:")
    # Do forward pass to initialize weights tracking
    output = layer(x_train)
    try:
        initial_sparsity = layer.calculate_sparsity_error('mul')
        weights = layer._last_train_O[0, 0, :2].detach().numpy()
        print(f"  Weights: [{weights[0]:.4f}, {weights[1]:.4f}]")
        print(f"  Sparsity error: {initial_sparsity:.6f}")
    except Exception as e:
        print(f"  Error: {e}")
    
    print(f"\nSimulating {10} training steps:")
    sparsity_values = []
    
    for step in range(10):
        optimizer.zero_grad()
        
        # Forward pass
        output = layer(x_train)
        
        # Calculate loss (MSE)
        loss = nn.MSELoss()(output, y_train)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate sparsity error after this training step
        try:
            sparsity = layer.calculate_sparsity_error('mul')
            sparsity_values.append(sparsity)
            weights = layer._last_train_O[0, 0, :2].detach().numpy()
            
            print(f"  Step {step+1:2d}: Loss={loss.item():.4f}, "
                  f"Weights=[{weights[0]:6.3f}, {weights[1]:6.3f}], "
                  f"Sparsity={sparsity:.6f}")
                  
        except Exception as e:
            print(f"  Step {step+1}: Error calculating sparsity: {e}")
    
    print(f"\nSparsity trend analysis:")
    if len(sparsity_values) >= 2:
        print(f"  Initial sparsity: {sparsity_values[0]:.6f}")  
        print(f"  Final sparsity: {sparsity_values[-1]:.6f}")
        print(f"  Change: {sparsity_values[-1] - sparsity_values[0]:+.6f}")
        
        # Check if sparsity is changing (not hardcoded)
        sparsity_range = max(sparsity_values) - min(sparsity_values)
        print(f"  Range: {sparsity_range:.6f}")
        
        if sparsity_range > 1e-6:
            print("  ✅ Sparsity values are changing - calculation appears to be based on actual weights")
        else:
            print("  ⚠️  Sparsity values are not changing - may be hardcoded or weights not updating")
    
    print("\nTesting with known discrete weights:")
    # Test with perfectly discrete weights [1.0, 1.0]
    with torch.no_grad():
        layer._last_train_O[0, 0, 0] = 1.0
        layer._last_train_O[0, 0, 1] = 1.0
    
    discrete_sparsity = layer.calculate_sparsity_error('mul')
    print(f"  Perfect discrete [1.0, 1.0]: sparsity = {discrete_sparsity:.6f} (should be 0.0)")
    
    # Test with worst-case weights [0.5, 0.5]  
    with torch.no_grad():
        layer._last_train_O[0, 0, 0] = 0.5
        layer._last_train_O[0, 0, 1] = 0.5
        
    worst_sparsity = layer.calculate_sparsity_error('mul')
    print(f"  Worst-case [0.5, 0.5]: sparsity = {worst_sparsity:.6f} (should be 0.5)")
    
    return sparsity_values

def test_sparsity_with_different_operations():
    """Test that sparsity calculation works with different operations."""
    
    print("\n=== Testing Sparsity with Different Operations ===")
    
    layer = DAGLayer(in_features=2, out_features=1, dag_depth=1, _enable_taps=False)
    
    # Initialize with forward pass
    x = torch.randn(2, 2)
    _ = layer(x)
    
    operations = ['add', 'sub', 'mul', 'div']
    for op in operations:
        try:
            sparsity = layer.calculate_sparsity_error(op)
            print(f"  {op.upper()}: sparsity = {sparsity:.6f}")
        except Exception as e:
            print(f"  {op.upper()}: Error = {e}")

def test_dag_depth_constraint():
    """Test that sparsity calculation correctly fails for dag_depth > 1."""
    
    print("\n=== Testing DAG Depth Constraint ===")
    
    for depth in [1, 2, 3]:
        try:
            layer = DAGLayer(in_features=2, out_features=1, dag_depth=depth, _enable_taps=False)
            x = torch.randn(2, 2)
            _ = layer(x)  # Initialize
            
            sparsity = layer.calculate_sparsity_error('mul')
            print(f"  dag_depth={depth}: sparsity = {sparsity:.6f} (should only work for depth=1)")
            
        except ValueError as e:
            print(f"  dag_depth={depth}: ✅ Correctly raised ValueError: {e}")
        except Exception as e:
            print(f"  dag_depth={depth}: ❌ Unexpected error: {e}")

if __name__ == "__main__":
    print("Verifying that sparsity error calculation is based on actual run data")
    print("=" * 70)
    
    # Run all tests
    sparsity_values = test_sparsity_during_simulated_training()
    test_sparsity_with_different_operations()
    test_dag_depth_constraint()
    
    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    
    if len(sparsity_values) > 0:
        sparsity_range = max(sparsity_values) - min(sparsity_values)
        if sparsity_range > 1e-6:
            print("✅ PASSED: Sparsity calculations are based on actual weight values")
        else:
            print("❌ FAILED: Sparsity values appear to be hardcoded or weights not changing")
    else:
        print("❌ FAILED: No sparsity values were calculated")