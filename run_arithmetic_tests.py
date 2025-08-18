#!/usr/bin/env python3

import sys
import traceback

from tests.test_dag_arithmetic import TestDAGArithmetic


def run_test(test_method, test_name):
    """Run a single test method and report results."""
    print(f"\n=== Running {test_name} ===")
    try:
        test_method()
        print(f"✓ {test_name} PASSED")
        return True
    except Exception as e:
        print(f"✗ {test_name} FAILED: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all arithmetic tests."""
    test_methods = [
        ("test_dag_addition_clamped", TestDAGArithmetic.test_dag_addition_clamped),
        ("test_dag_addition_unclamped", TestDAGArithmetic.test_dag_addition_unclamped),
        (
            "test_dag_subtraction_clamped",
            TestDAGArithmetic.test_dag_subtraction_clamped,
        ),
        (
            "test_dag_subtraction_unclamped",
            TestDAGArithmetic.test_dag_subtraction_unclamped,
        ),
        (
            "test_dag_multiplication_clamped",
            TestDAGArithmetic.test_dag_multiplication_clamped,
        ),
        (
            "test_dag_multiplication_unclamped",
            TestDAGArithmetic.test_dag_multiplication_unclamped,
        ),
        ("test_dag_division_clamped", TestDAGArithmetic.test_dag_division_clamped),
        ("test_dag_division_unclamped", TestDAGArithmetic.test_dag_division_unclamped),
    ]

    tests = []
    for test_name, test_method in test_methods:
        test_instance = TestDAGArithmetic()
        test_instance.setup_method()
        bound_method = test_method.__get__(test_instance, TestDAGArithmetic)
        tests.append((bound_method, test_name))

    passed = 0
    total = len(tests)

    for test_method, test_name in tests:
        if run_test(test_method, test_name):
            passed += 1

    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    print(f"Failed: {total - passed}/{total}")

    if passed == total:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
