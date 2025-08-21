#!/usr/bin/env python3
"""
Tiny e2e test to verify training still works after DAG changes.
"""

import subprocess
import sys


def run_tiny_training():
    """Run a very small training session to verify nothing is broken."""
    cmd = [
        sys.executable,
        "experiments/single_layer_benchmark.py",
        "--layer-type",
        "DAG",
        "--note",
        "tiny_e2e_test",
        "--operation",
        "add",
        "--input-size",
        "2",
        "--batch-size",
        "2",
        "--max-iterations",
        "10",  # Very short
        "--learning-rate",
        "1e-3",
        "--log-interval",
        "10",
        "--interpolation-range",
        "[-2.0,2.0]",
        "--extrapolation-range",
        "[[-6.0,-2.0],[2.0,6.0]]",
        "--seed",
        "0",
        "--no-cuda",
        "--lr-cosine",
        "--lr-min",
        "1e-4",
        "--clip-grad-norm",
        "1.0",
        "--no-open-browser",  # Don't open browser for test
    ]

    print("Running tiny e2e training test...")
    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode == 0:
            print("✅ Tiny training test PASSED")
            print("Training completed without errors")
            return True
        else:
            print("❌ Tiny training test FAILED")
            print(f"Return code: {result.returncode}")
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("❌ Tiny training test FAILED - timeout after 120s")
        return False
    except Exception as e:
        print(f"❌ Tiny training test FAILED - exception: {e}")
        return False


if __name__ == "__main__":
    success = run_tiny_training()
    sys.exit(0 if success else 1)
