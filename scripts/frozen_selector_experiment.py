#!/usr/bin/env python3
"""
Comprehensive frozen selector experiment for arithmetic operations.

Tests grokking success rates with frozen O selectors across different seeds 
and interpolation ranges from the paper to understand how selector freezing
affects learning.

Experiment design:
1. Test mul/add with freeze_O_selector_mul=True 
2. Test sub/div with freeze_O_selectors_div=True
3. Use linear_biased_init_G=True for both
4. Test across multiple seeds and all paper ranges
"""

import argparse
import json
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# Test seeds - diverse set to explore initialization space
TEST_SEEDS = [122, 223, 42, 777, 1337]

# Paper interpolation/extrapolation ranges
PAPER_RANGES = [
    ([-20, -10], [-40, -20]),
    ([-2, -1], [-6, -2]),
    ([-1.2, -1.1], [-6.1, -1.2]),
    ([-0.2, -0.1], [-2, -0.2]),
    ([-2, 2], [[-6, -2], [2, 6]]),
    ([0.1, 0.2], [0.2, 2]),
    ([1, 2], [2, 6]),
    ([1.1, 1.2], [1.2, 6]),
    ([10, 20], [20, 40]),
]

# Base hyperparameters 
BASE_HYPERPARAMS = {
    "layer_type": "DAG",
    "input_size": 2,
    "batch_size": 512,
    "max_iterations": 3000,
    "learning_rate": 1e-2,  # Higher LR needed for grokking with frozen selectors
    "no_cuda": True,
    "log_interval": 100,
    "no_open_browser": True,
}

# Grokking detection
GROKKING_THRESHOLD = 1e-8
SUCCESS_EARLY_STOP_PATTERN = "Early stopping at step"


def run_single_experiment(operation: str, seed: int, interp_range, extrap_range, frozen_config: dict):
    """Run a single experiment for specific operation, seed, and range."""
    
    start_time = time.time()
    
    try:
        cmd = ["python", "experiments/single_layer_benchmark.py"]
        
        # Add base hyperparameters
        for param, value in BASE_HYPERPARAMS.items():
            if param in ["no_cuda", "no_open_browser"]:
                if value:
                    cmd.append(f'--{param.replace("_", "-")}')
            else:
                cmd.extend([f'--{param.replace("_", "-")}', str(value)])
        
        # Add operation and seed
        cmd.extend(["--operation", operation])
        cmd.extend(["--seed", str(seed)])
        
        # Add ranges
        cmd.extend(["--interpolation-range", str(interp_range)])
        cmd.extend(["--extrapolation-range", str(extrap_range)])
        
        print(f"Running {operation} seed {seed} range {interp_range} -> {extrap_range}")
        print(f"Frozen config: {frozen_config}")
        
        # Run the experiment
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=600,  # 10 minute timeout
            cwd="/Users/paul_curry/ai2/nalm-benchmark"
        )
        
        duration = time.time() - start_time
        output_lines = result.stdout.split('\n') if result.stdout else []
        
        # Parse results
        early_stopped = False
        grokked = False  
        grok_step = None
        final_inter_loss = float("inf")
        final_extra_loss = float("inf")
        
        # Check for early stopping
        for line in output_lines:
            if SUCCESS_EARLY_STOP_PATTERN in line:
                early_stopped = True
                try:
                    grok_step = int(line.split("step ")[1].split(":")[0])
                    grokked = True
                except (ValueError, IndexError):
                    grokked = False
                break
        
        # If no early stopping, check final performance
        if not early_stopped:
            for line in reversed(output_lines):
                if line.strip().startswith("- loss_valid_inter:"):
                    try:
                        final_inter_loss = float(line.split(":")[1].strip())
                        if final_inter_loss < GROKKING_THRESHOLD:
                            grokked = True
                        break
                    except (ValueError, IndexError):
                        continue
                elif line.strip().startswith("- loss_valid_extra:"):
                    try:
                        final_extra_loss = float(line.split(":")[1].strip())
                    except (ValueError, IndexError):
                        continue
        
        success = result.returncode == 0
        
        return {
            "operation": operation,
            "seed": seed,
            "interp_range": interp_range,
            "extrap_range": extrap_range,
            "frozen_config": frozen_config,
            "success": success,
            "grokked": grokked,
            "early_stopped": early_stopped,
            "grok_step": grok_step,
            "duration": duration,
            "final_inter_loss": final_inter_loss,
            "final_extra_loss": final_extra_loss,
            "stdout_excerpt": result.stdout[-2000:] if result.stdout else "",
            "stderr_excerpt": result.stderr[-1000:] if result.stderr else "",
        }
        
    except subprocess.TimeoutExpired:
        return {
            "operation": operation,
            "seed": seed,
            "interp_range": interp_range,
            "extrap_range": extrap_range,
            "frozen_config": frozen_config,
            "success": False,
            "grokked": False,
            "early_stopped": False,
            "grok_step": None,
            "duration": time.time() - start_time,
            "final_inter_loss": float("inf"),
            "final_extra_loss": float("inf"),
            "stdout_excerpt": "TIMEOUT",
            "stderr_excerpt": "TIMEOUT",
        }
    except Exception as e:
        return {
            "operation": operation,
            "seed": seed,
            "interp_range": interp_range,
            "extrap_range": extrap_range,
            "frozen_config": frozen_config,
            "success": False,
            "grokked": False,
            "early_stopped": False,
            "grok_step": None,
            "duration": time.time() - start_time,
            "final_inter_loss": float("inf"),
            "final_extra_loss": float("inf"),
            "stdout_excerpt": "",
            "stderr_excerpt": str(e),
        }


def analyze_results(results):
    """Analyze the results and compute success rates."""
    
    # Group by operation and configuration
    by_operation = {}
    for result in results:
        key = f"{result['operation']}"
        if key not in by_operation:
            by_operation[key] = []
        by_operation[key].append(result)
    
    analysis = {
        "total_experiments": len(results),
        "successful_runs": sum(1 for r in results if r["success"]),
        "grokked_runs": sum(1 for r in results if r["grokked"]),
        "operations": {},
    }
    
    for operation, op_results in by_operation.items():
        total_tests = len(op_results)
        successful_runs = [r for r in op_results if r["success"]]
        grokked_runs = [r for r in op_results if r["grokked"]]
        
        # Calculate rates
        success_rate = len(successful_runs) / total_tests if total_tests > 0 else 0
        grok_rate = len(grokked_runs) / total_tests if total_tests > 0 else 0
        
        # Calculate average grok step
        grok_steps = [r["grok_step"] for r in grokked_runs if r["grok_step"] is not None]
        avg_grok_step = np.mean(grok_steps) if grok_steps else None
        median_grok_step = np.median(grok_steps) if grok_steps else None
        
        # Group by range for detailed analysis
        by_range = {}
        for result in op_results:
            range_key = f"{result['interp_range']}"
            if range_key not in by_range:
                by_range[range_key] = []
            by_range[range_key].append(result)
        
        range_analysis = {}
        for range_key, range_results in by_range.items():
            range_total = len(range_results)
            range_grokked = sum(1 for r in range_results if r["grokked"])
            range_analysis[range_key] = {
                "total_tests": range_total,
                "grokked": range_grokked,
                "grok_rate": range_grokked / range_total if range_total > 0 else 0,
                "avg_grok_step": np.mean([r["grok_step"] for r in range_results if r["grok_step"] is not None]) if any(r["grok_step"] for r in range_results) else None
            }
        
        analysis["operations"][operation] = {
            "total_tests": total_tests,
            "successful_runs": len(successful_runs),
            "grokked_runs": len(grokked_runs),
            "success_rate": success_rate,
            "grok_rate": grok_rate,
            "avg_grok_step": avg_grok_step,
            "median_grok_step": median_grok_step,
            "by_range": range_analysis,
        }
    
    return analysis


def print_summary(analysis):
    """Print a summary of the analysis."""
    
    print("\n" + "="*80)
    print("FROZEN SELECTOR EXPERIMENT RESULTS")
    print("="*80)
    
    print(f"Total experiments: {analysis['total_experiments']}")
    print(f"Successful runs: {analysis['successful_runs']}")
    print(f"Grokked runs: {analysis['grokked_runs']}")
    print(f"Overall grok rate: {analysis['grokked_runs']/analysis['total_experiments']:.1%}")
    
    print(f"\nPer Operation Summary:")
    print("-" * 60)
    
    for op, stats in analysis["operations"].items():
        print(f"\n{op.upper()}:")
        print(f"  Success rate: {stats['success_rate']:.1%} ({stats['successful_runs']}/{stats['total_tests']})")
        print(f"  Grok rate: {stats['grok_rate']:.1%} ({stats['grokked_runs']}/{stats['total_tests']})")
        if stats['avg_grok_step']:
            print(f"  Avg grok step: {stats['avg_grok_step']:.0f}")
            print(f"  Median grok step: {stats['median_grok_step']:.0f}")
        
        print(f"  Range breakdown:")
        for range_key, range_stats in stats["by_range"].items():
            print(f"    {range_key}: {range_stats['grok_rate']:.1%} ({range_stats['grokked']}/{range_stats['total_tests']})")


def main():
    parser = argparse.ArgumentParser(description="Run frozen selector experiment")
    parser.add_argument("--max-workers", type=int, default=2, help="Maximum parallel workers")
    parser.add_argument("--output-dir", type=str, default="experiment_results", help="Output directory")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Generate all experiment configurations
    experiments = []
    
    # Group 1: mul/add with freeze_O_selector_mul
    for operation in ["mul", "add"]:
        frozen_config = {"freeze_O_selector_mul": True, "freeze_O_selectors_div": False}
        for seed in TEST_SEEDS:
            for interp_range, extrap_range in PAPER_RANGES:
                experiments.append((operation, seed, interp_range, extrap_range, frozen_config))
    
    # Group 2: sub/div with freeze_O_selectors_div  
    for operation in ["sub", "div"]:
        frozen_config = {"freeze_O_selectors_div": True, "freeze_O_selector_mul": False}
        for seed in TEST_SEEDS:
            for interp_range, extrap_range in PAPER_RANGES:
                experiments.append((operation, seed, interp_range, extrap_range, frozen_config))
    
    print(f"Running {len(experiments)} total experiments...")
    print(f"Operations: mul/add (with freeze_O_selector_mul), sub/div (with freeze_O_selectors_div)")
    print(f"Seeds: {TEST_SEEDS}")
    print(f"Ranges: {len(PAPER_RANGES)} paper ranges")
    print(f"Max workers: {args.max_workers}")
    
    results = []
    
    # Run experiments in parallel
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_exp = {
            executor.submit(run_single_experiment, *exp): exp 
            for exp in experiments
        }
        
        for future in as_completed(future_to_exp):
            result = future.result()
            results.append(result)
            
            # Print progress
            if result["grokked"]:
                print(f"✅ {result['operation']} seed {result['seed']} range {result['interp_range']}: GROKKED at step {result['grok_step']}")
            else:
                print(f"❌ {result['operation']} seed {result['seed']} range {result['interp_range']}: Failed (loss: {result['final_inter_loss']:.2e})")
    
    # Analyze results
    analysis = analyze_results(results)
    
    # Save results
    timestamp = int(time.time())
    results_file = output_dir / f"frozen_selector_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "experiment_config": {
                "seeds": TEST_SEEDS,
                "ranges": PAPER_RANGES,
                "base_hyperparams": BASE_HYPERPARAMS,
            },
            "results": results,
            "analysis": analysis
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Print summary
    print_summary(analysis)


if __name__ == "__main__":
    main()