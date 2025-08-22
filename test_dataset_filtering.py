#!/usr/bin/env python3

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from stable_nalu.dataset.simple_function_static import \
    SimpleFunctionStaticDataset


def test_division_filtering():
    print("Testing division dataset filtering...")

    # Test without filtering (should get some extreme values)
    print("\n1. Testing without filtering:")
    dataset_unfiltered = SimpleFunctionStaticDataset(
        operation="div",
        input_size=2,
        subset_ratio=0.5,
        num_subsets=2,
        seed=42,
        max_result_magnitude=None,  # No filtering
    )

    # Sample some data
    train_data = dataset_unfiltered.fork(sample_range=[-2.0, 2.0])

    # Check targets
    extreme_count = 0
    total_count = 20
    targets = []

    for i in range(total_count):
        x, t = train_data[i]
        targets.append(float(t.item()))
        if abs(t.item()) > 50:
            extreme_count += 1

    print(f"  Sample targets: {targets[:10]}")
    print(f"  Max target magnitude: {max(abs(t) for t in targets):.2f}")
    print(f"  Extreme targets (>50): {extreme_count}/{total_count}")

    # Test with filtering (should have no extreme values)
    print("\n2. Testing with filtering (max_target=10):")
    dataset_filtered = SimpleFunctionStaticDataset(
        operation="div",
        input_size=2,
        subset_ratio=0.5,
        num_subsets=2,
        seed=42,
        max_result_magnitude=10.0,  # Filter targets > 10
    )

    train_data_filtered = dataset_filtered.fork(sample_range=[-2.0, 2.0])

    # Check filtered targets
    extreme_count_filtered = 0
    targets_filtered = []

    for i in range(total_count):
        x, t = train_data_filtered[i]
        targets_filtered.append(float(t.item()))
        if abs(t.item()) > 10:
            extreme_count_filtered += 1

    print(f"  Sample targets: {targets_filtered[:10]}")
    print(f"  Max target magnitude: {max(abs(t) for t in targets_filtered):.2f}")
    print(f"  Extreme targets (>10): {extreme_count_filtered}/{total_count}")

    # Validation
    if extreme_count_filtered == 0:
        print("  ✅ Filtering works correctly!")
    else:
        print("  ❌ Filtering failed - still has extreme targets")

    return extreme_count_filtered == 0


def test_other_operations():
    print("\n3. Testing that filtering doesn't break other operations:")

    for op in ["add", "mul", "sub"]:
        dataset = SimpleFunctionStaticDataset(
            operation=op,
            input_size=2,
            subset_ratio=0.5,
            num_subsets=2,
            seed=42,
            max_result_magnitude=10.0,
        )

        train_data = dataset.fork(sample_range=[-2.0, 2.0])
        x, t = train_data[0]
        print(f"  {op}: target = {t.item():.3f} ✅")


if __name__ == "__main__":
    success = test_division_filtering()
    test_other_operations()

    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)
