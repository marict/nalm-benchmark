#!/usr/bin/env python3
"""
Comprehensive frozen selector experiment with all seeds and key ranges.
"""

import subprocess
import time
import json
from pathlib import Path

# Test configuration
TEST_SEEDS = [122, 223, 42, 777, 1337]

# Key paper ranges for comprehensive testing
KEY_RANGES = [
    ([-2, 2], [[-6, -2], [2, 6]]),      # Standard good range
    ([-2, -1], [-6, -2]),               # Challenging negative range  
    ([1, 2], [2, 6]),                   # Positive range
    ([-0.2, -0.1], [-2, -0.2]),        # Small negative range
    ([0.1, 0.2], [0.2, 2]),             # Small positive range
]

def modify_dag_defaults(freeze_div=False, freeze_mul=False):
    """Modify DAG defaults by editing the file directly."""
    dag_file = "/Users/paul_curry/ai2/nalm-benchmark/stable_nalu/layer/dag.py"
    
    with open(dag_file, 'r') as f:
        content = f.read()
    
    if freeze_div and not freeze_mul:
        content = content.replace(
            "freeze_O_selectors_div: bool = False,",
            "freeze_O_selectors_div: bool = True,"
        ).replace(
            "freeze_O_selector_mul: bool = True,", 
            "freeze_O_selector_mul: bool = False,"
        )
    elif freeze_mul and not freeze_div:
        content = content.replace(
            "freeze_O_selectors_div: bool = True,",
            "freeze_O_selectors_div: bool = False,"
        ).replace(
            "freeze_O_selector_mul: bool = False,",
            "freeze_O_selector_mul: bool = True,"
        )
    
    with open(dag_file, 'w') as f:
        f.write(content)

def run_experiment(operation, seed, interp_range, extrap_range):
    """Run a single experiment."""
    cmd = [
        "python", "experiments/single_layer_benchmark.py",
        "--layer-type", "DAG", "--no-open-browser",
        "--operation", operation, "--seed", str(seed),
        "--input-size", "2", "--batch-size", "512", 
        "--max-iterations", "3000", "--learning-rate", "1e-2",
        "--interpolation-range", str(interp_range),
        "--extrapolation-range", str(extrap_range),
        "--no-cuda", "--log-interval", "100"
    ]
    
    print(f"Testing {operation} seed {seed} range {interp_range}")
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        duration = time.time() - start_time
        
        output_lines = result.stdout.split('\n') if result.stdout else []
        
        grokked = False
        grok_step = None
        final_inter_loss = float("inf")
        
        # Check for early stopping
        for line in output_lines:
            if "Early stopping at step" in line:
                grokked = True
                try:
                    grok_step = int(line.split("step ")[1].split(":")[0])
                except:
                    pass
                break
        
        # Check final loss if no early stopping
        if not grokked:
            for line in reversed(output_lines):
                if "- loss_valid_inter:" in line:
                    try:
                        final_inter_loss = float(line.split(":")[1].strip())
                        if final_inter_loss < 1e-8:
                            grokked = True
                        break
                    except:
                        continue
        
        return {
            "operation": operation,
            "seed": seed,
            "interp_range": interp_range,
            "extrap_range": extrap_range,
            "success": result.returncode == 0,
            "grokked": grokked,
            "grok_step": grok_step,
            "duration": duration,
            "final_inter_loss": final_inter_loss,
        }
        
    except subprocess.TimeoutExpired:
        return {
            "operation": operation,
            "seed": seed, 
            "interp_range": interp_range,
            "extrap_range": extrap_range,
            "success": False,
            "grokked": False,
            "grok_step": None,
            "duration": 180,
            "final_inter_loss": float("inf"),
        }

def main():
    results = []
    
    print("=" * 80)
    print("COMPREHENSIVE FROZEN SELECTOR EXPERIMENT")
    print("=" * 80)
    print(f"Seeds: {TEST_SEEDS}")
    print(f"Ranges: {len(KEY_RANGES)} key ranges")
    total_experiments = len(TEST_SEEDS) * len(KEY_RANGES) * 4  # 4 operations
    print(f"Total experiments: {total_experiments}")
    
    completed = 0
    
    # Group 1: mul/add with freeze_O_selector_mul
    print(f"\\n🔄 Setting up for MUL/ADD (freeze_O_selector_mul=True)...")
    modify_dag_defaults(freeze_div=False, freeze_mul=True)
    
    for operation in ["mul", "add"]:
        print(f"\\n--- Testing {operation.upper()} ---")
        for seed in TEST_SEEDS:
            for interp_range, extrap_range in KEY_RANGES:
                result = run_experiment(operation, seed, interp_range, extrap_range)
                results.append(result)
                completed += 1
                
                if result["grokked"]:
                    print(f"✅ {operation} seed {seed} {interp_range}: GROKKED at step {result['grok_step']}")
                else:
                    print(f"❌ {operation} seed {seed} {interp_range}: Failed (loss: {result['final_inter_loss']:.2e})")
                
                print(f"   Progress: {completed}/{total_experiments} ({completed/total_experiments*100:.1f}%)")
    
    # Group 2: sub/div with freeze_O_selectors_div
    print(f"\\n🔄 Setting up for SUB/DIV (freeze_O_selectors_div=True)...")
    modify_dag_defaults(freeze_div=True, freeze_mul=False)
    
    for operation in ["sub", "div"]:
        print(f"\\n--- Testing {operation.upper()} ---")
        for seed in TEST_SEEDS:
            for interp_range, extrap_range in KEY_RANGES:
                result = run_experiment(operation, seed, interp_range, extrap_range)
                results.append(result)
                completed += 1
                
                if result["grokked"]:
                    print(f"✅ {operation} seed {seed} {interp_range}: GROKKED at step {result['grok_step']}")
                else:
                    print(f"❌ {operation} seed {seed} {interp_range}: Failed (loss: {result['final_inter_loss']:.2e})")
                
                print(f"   Progress: {completed}/{total_experiments} ({completed/total_experiments*100:.1f}%)")
    
    # Analyze results
    print(f"\\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    
    by_operation = {}
    for result in results:
        op = result["operation"]
        if op not in by_operation:
            by_operation[op] = []
        by_operation[op].append(result)
    
    total_experiments = len(results)
    total_grokked = sum(1 for r in results if r["grokked"])
    
    print(f"Total experiments: {total_experiments}")
    print(f"Total grokked: {total_grokked}")
    print(f"Overall success rate: {total_grokked/total_experiments:.1%}")
    
    print(f"\\nPer-operation breakdown:")
    for op, op_results in by_operation.items():
        grokked_count = sum(1 for r in op_results if r["grokked"])
        total_count = len(op_results)
        success_rate = grokked_count / total_count if total_count > 0 else 0
        
        avg_grok_step = None
        if grokked_count > 0:
            grok_steps = [r["grok_step"] for r in op_results if r["grok_step"] is not None]
            if grok_steps:
                avg_grok_step = sum(grok_steps) / len(grok_steps)
        
        print(f"{op.upper()}: {success_rate:.1%} ({grokked_count}/{total_count})" + 
              (f", avg grok step: {avg_grok_step:.0f}" if avg_grok_step else ""))
    
    # Save results
    timestamp = int(time.time())
    output_dir = Path("experiment_results")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"comprehensive_frozen_results_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "config": {
                "seeds": TEST_SEEDS,
                "ranges": KEY_RANGES,
                "learning_rate": "1e-2",
                "max_iterations": 3000
            },
            "results": results
        }, f, indent=2)
    
    print(f"\\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()