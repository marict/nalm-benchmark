#!/usr/bin/env python3
"""
Test numerical limits for division using the full training pipeline.
This uses the actual single_layer_benchmark.py with custom DAG parameters.
"""

import os
import subprocess
import sys
import tempfile
import time


def run_division_with_limits(mag_min, mag_max, log_lim_offset, test_name, seed=42):
    """Run division training with custom numerical limits."""

    # We need to create a custom DAG configuration
    # Since we can't pass these as command-line args, we'll modify the imports temporarily

    # Create a temporary script that imports and runs with custom limits
    temp_script = f"""
import sys
sys.path.insert(0, '/Users/paul_curry/ai2/nalm-benchmark')

import torch
import torch.nn as nn
from stable_nalu.layer.dag import DAGLayer

# Monkey patch the DAGLayer to use custom limits
original_init = DAGLayer.__init__

def patched_init(self, *args, **kwargs):
    # Force our custom limits
    kwargs['mag_min'] = {{mag_min}}
    kwargs['mag_max'] = {{mag_max}} 
    kwargs['log_lim_offset'] = {{log_lim_offset}}
    return original_init(self, *args, **kwargs)

DAGLayer.__init__ = patched_init

# Now run the standard training
import subprocess
cmd = [
    sys.executable, 
    '/Users/paul_curry/ai2/nalm-benchmark/experiments/single_layer_benchmark.py',
    '--layer-type', 'DAG',
    '--operation', 'div',
    '--input-size', '2',
    '--batch-size', '128',
    '--max-iterations', '1500',
    '--learning-rate', '1e-3',
    '--log-interval', '1500',
    '--interpolation-range', '[-2.0,2.0]',
    '--extrapolation-range', '[[-6.0,-2.0],[2.0,6.0]]',
    '--seed', '{{seed}}',
    '--no-cuda',
    '--no-open-browser',
    '--no-save',
    '--lr-cosine',
    '--lr-min', '1e-4',
    '--clip-grad-norm', '1.0',
    '--note', 'limits_test_{{test_name}}'
]

result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
print("RESULT_START")
print(f"RETURNCODE: {{result.returncode}}")
print(f"STDOUT: {{result.stdout}}")
print(f"STDERR: {{result.stderr}}")
print("RESULT_END")
"""

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(temp_script)
            temp_file = f.name

        print(f"🔧 Testing {test_name}")
        print(
            f"   mag_min={mag_min:.0e}, mag_max={mag_max:.0e}, log_lim_offset={log_lim_offset}"
        )
        print(f"   ", end="", flush=True)

        start_time = time.time()
        result = subprocess.run(
            [sys.executable, temp_file], capture_output=True, text=True, timeout=120
        )
        duration = time.time() - start_time

        # Clean up temp file
        os.unlink(temp_file)

        if result.returncode != 0:
            print(f"❌ SCRIPT ERROR ({duration:.1f}s)")
            return {"success": False, "error": "script_error", "duration": duration}

        # Parse the embedded result
        output = result.stdout
        if "RESULT_START" not in output:
            print(f"❌ NO RESULT ({duration:.1f}s)")
            return {"success": False, "error": "no_result", "duration": duration}

        result_section = output.split("RESULT_START")[1].split("RESULT_END")[0]
        lines = result_section.strip().split("\n")

        returncode = None
        training_output = ""

        for line in lines:
            if line.startswith("RETURNCODE:"):
                returncode = int(line.split(": ")[1])
            elif line.startswith("STDOUT:"):
                training_output = line.split(": ", 1)[1] if ": " in line else ""

        if returncode != 0:
            print(f"❌ TRAINING FAILED ({duration:.1f}s)")
            # Print stderr to see what went wrong
            training_stderr = ""
            for line in lines:
                if line.startswith("STDERR:"):
                    training_stderr = line.split(": ", 1)[1] if ": " in line else ""
                    break
            if training_stderr:
                print(f"    Error: {training_stderr[:100]}")
            return {"success": False, "error": "training_failed", "duration": duration}

        # Parse training results
        final_error = None
        grokked = "Early stopping" in training_output

        # Extract final interpolation error
        lines = training_output.split("\\n")
        for line in reversed(lines):
            if "inter:" in line and "train" in line:
                try:
                    inter_part = line.split("inter: ")[1].split(",")[0]
                    final_error = float(inter_part)
                    break
                except (IndexError, ValueError):
                    continue

        if grokked:
            print(f"🎉 GROKKED! Error: {final_error:.2e} ({duration:.1f}s)")
        else:
            print(f"❌ No grok. Error: {final_error:.2e} ({duration:.1f}s)")

        return {
            "success": True,
            "final_error": final_error,
            "grokked": grokked,
            "duration": duration,
        }

    except subprocess.TimeoutExpired:
        print(f"❌ TIMEOUT")
        if "temp_file" in locals():
            try:
                os.unlink(temp_file)
            except:
                pass
        return {"success": False, "error": "timeout", "duration": 120.0}
    except Exception as e:
        print(f"❌ ERROR: {str(e)[:50]}")
        if "temp_file" in locals():
            try:
                os.unlink(temp_file)
            except:
                pass
        return {"success": False, "error": str(e), "duration": 0.0}


def main():
    """Test numerical limits with full training pipeline."""
    print("Division Numerical Limits - Full Training Test")
    print("=" * 50)

    # Focus on the most promising configurations from the simple test
    test_configs = [
        # Baseline (current defaults)
        (1e-11, 1e6, 1.0, "baseline"),
        # Best performers from simple test
        (1e-11, 1e6, 0.1, "tiny_log_offset"),
        (1e-8, 1e4, 1.0, "tighter_mag_range"),
        # Division-specific optimizations
        (1e-9, 1e5, 0.5, "division_optimized"),
        (1e-8, 1e3, 0.3, "ultra_stable"),
        # More extreme variants
        (1e-10, 1e4, 0.8, "moderate_tight"),
        (1e-12, 1e8, 1.5, "looser_variant"),
    ]

    print(f"Testing {len(test_configs)} configurations with full training pipeline...")

    results = []

    for mag_min, mag_max, log_lim_offset, name in test_configs:
        result = run_division_with_limits(mag_min, mag_max, log_lim_offset, name)
        results.append((name, mag_min, mag_max, log_lim_offset, result))

    # Analysis
    print(f"\n📊 RESULTS SUMMARY:")
    print(f"=" * 50)

    successful_results = [
        (name, params, res)
        for name, *params, res in results
        if res.get("success", False)
    ]

    if successful_results:
        # Check for any grokking
        grokked_configs = [
            (name, params, res)
            for name, params, res in successful_results
            if res.get("grokked", False)
        ]

        if grokked_configs:
            print(f"🎉 BREAKTHROUGH! Division grokked with:")
            for name, (mag_min, mag_max, log_lim_offset), res in grokked_configs:
                print(
                    f"   {name}: mag_min={mag_min:.0e}, mag_max={mag_max:.0e}, log_lim_offset={log_lim_offset}"
                )
                print(
                    f"     Final error: {res['final_error']:.2e}, Duration: {res['duration']:.1f}s"
                )

            return True

        # Sort by final error
        successful_results.sort(key=lambda x: x[2]["final_error"] or float("inf"))

        print(f"No grokking, but ranking by final error:")
        for i, (name, (mag_min, mag_max, log_lim_offset), res) in enumerate(
            successful_results
        ):
            error = res["final_error"]
            print(f"   {i+1}. {name}: {error:.2e}")
            if i == 0:  # Show params for best
                print(
                    f"      mag_min={mag_min:.0e}, mag_max={mag_max:.0e}, log_lim_offset={log_lim_offset}"
                )

        # Compare to baseline
        baseline_res = next(
            (res for name, params, res in results if name == "baseline"), None
        )
        if baseline_res and baseline_res.get("success"):
            best_res = successful_results[0][2]
            baseline_error = baseline_res["final_error"]
            best_error = best_res["final_error"]

            if best_error < baseline_error:
                improvement = baseline_error / best_error
                print(f"\n✨ Best config is {improvement:.1f}x better than baseline!")
            else:
                print(f"\n🔍 No significant improvement over baseline.")

    else:
        print(f"❌ All configurations failed!")

    return False


if __name__ == "__main__":
    success = main()

    if success:
        print(f"\n🏆 SUCCESS: Found numerical limits that enable division grokking!")
    else:
        print(f"\n🔬 No breakthrough found with numerical limit adjustments.")
        print(f"   The division training issue likely requires architectural changes.")

    sys.exit(0 if success else 1)
