#!/usr/bin/env python3
"""
Chunked comprehensive frozen selector test - runs one operation at a time and saves incrementally.
"""

import subprocess
import time
import json
from pathlib import Path
import pandas as pd
import sys
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading

# Test configuration
TEST_SEEDS = [122, 223, 42, 777, 1337]
TEST_RANGES = [
    ([-2, 2], [[-6, -2], [2, 6]], "standard"),
    ([-2, -1], [-6, -2], "neg_moderate"),
    ([1, 2], [2, 6], "pos_moderate"),
    ([-1.2, -1.1], [-6.1, -1.2], "neg_narrow"),
    ([0.1, 0.2], [0.2, 2], "pos_small"),
    ([-0.2, -0.1], [-2, -0.2], "neg_small"),
    ([1.1, 1.2], [1.2, 6], "pos_narrow"),
    ([-20, -10], [-40, -20], "neg_large"),
    ([10, 20], [20, 40], "pos_large"),
]

OPERATIONS = ["mul", "add", "sub", "div"]

def run_single_test(operation, seed, interp_range, extrap_range):
    """Run a single test and return result."""
    
    # Base command with updated hyperparameters
    cmd = [
        "python", "experiments/single_layer_benchmark.py",
        "--layer-type", "DAG", "--no-open-browser",
        "--operation", operation, "--seed", str(seed),
        "--input-size", "2", "--batch-size", "128", 
        "--max-iterations", "2000", "--learning-rate", "1e-3",
        "--interpolation-range", str(interp_range),
        "--extrapolation-range", str(extrap_range),
        "--no-cuda", "--log-interval", "100", "--clip-grad-norm", "0.01"
    ]
    
    # Add frozen selector arguments based on operation
    if operation in ["mul", "add"]:
        cmd.append("--freeze-O-mul")
        frozen_config = "freeze_O_mul=True"
    elif operation in ["sub", "div"]:
        cmd.append("--freeze-O-div")
        frozen_config = "freeze_O_div=True"
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
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
            "frozen_config": frozen_config,
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
            "frozen_config": frozen_config,
            "grokked": False,
            "grok_step": None,
            "duration": 120,
            "final_inter_loss": float("inf"),
        }

def load_existing_results():
    """Load existing results if available."""
    output_dir = Path("experiment_results")
    output_dir.mkdir(exist_ok=True)
    
    results_file = output_dir / "chunked_comprehensive_results.json"
    if results_file.exists():
        with open(results_file, 'r') as f:
            data = json.load(f)
            return data.get("results", [])
    return []

def save_results(results):
    """Save results incrementally."""
    timestamp = int(time.time())
    output_dir = Path("experiment_results")
    output_dir.mkdir(exist_ok=True)
    results_file = output_dir / "chunked_comprehensive_results.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "config": {
                "seeds": TEST_SEEDS,
                "ranges": [(ir, er, name) for ir, er, name in TEST_RANGES],
                "operations": OPERATIONS
            },
            "results": results
        }, f, indent=2)

def run_operation_chunk(target_operation, target_range_idx=None):
    """Run tests for a specific operation and optionally a specific range."""
    results = load_existing_results()
    
    if target_range_idx is not None:
        ranges_to_test = [TEST_RANGES[target_range_idx]]
        print(f"🔸 Testing {target_operation.upper()} on range {target_range_idx}: {TEST_RANGES[target_range_idx][2]}")
    else:
        ranges_to_test = TEST_RANGES
        print(f"🔸 Testing all ranges for {target_operation.upper()}")
    
    for range_idx, (interp_range, extrap_range, range_name) in enumerate(ranges_to_test):
        if target_range_idx is not None:
            range_idx = target_range_idx
            
        print(f"  Range {range_name}: {interp_range} → {extrap_range}")
        print(f"  {target_operation.upper()}: ", end="")
        
        for seed in TEST_SEEDS:
            # Check if this result already exists
            existing = [r for r in results 
                       if r["operation"] == target_operation 
                       and r["seed"] == seed 
                       and str(r["interp_range"]) == str(interp_range)]
            
            if existing:
                result = existing[0]
                print(f"✅" if result["grokked"] else "❌", end="")
                print(f"({seed}-cached)", end=" ")
                continue
                
            result = run_single_test(target_operation, seed, interp_range, extrap_range)
            results.append(result)
            
            if result["grokked"]:
                print("✅", end="")
            else:
                print("❌", end="")
            
            print(f"({seed})", end=" ")
            
            # Save after each test
            save_results(results)
        
        # Calculate success rate for this operation/range combo
        op_range_results = [r for r in results 
                           if r["operation"] == target_operation 
                           and str(r["interp_range"]) == str(interp_range)]
        success_rate = sum(1 for r in op_range_results if r["grokked"]) / len(op_range_results) * 100
        print(f"→ {success_rate:.0f}%")

def run_test_batch(test_configs):
    """Helper function to run a batch of tests - for multiprocessing."""
    batch_results = []
    for config in test_configs:
        result = run_single_test(config['operation'], config['seed'], config['interp_range'], config['extrap_range'])
        batch_results.append(result)
    return batch_results

def run_operation_chunk_concurrent(target_operation, target_range_idx=None, max_workers=4):
    """Run tests for a specific operation concurrently."""
    results = load_existing_results()
    
    if target_range_idx is not None:
        ranges_to_test = [TEST_RANGES[target_range_idx]]
        print(f"🔸 Testing {target_operation.upper()} on range {target_range_idx}: {TEST_RANGES[target_range_idx][2]} (CONCURRENT)")
    else:
        ranges_to_test = TEST_RANGES
        print(f"🔸 Testing all ranges for {target_operation.upper()} (CONCURRENT - {max_workers} workers)")
    
    # Collect all test configurations needed
    test_configs = []
    for interp_range, extrap_range, range_name in ranges_to_test:
        for seed in TEST_SEEDS:
            # Check if this result already exists
            existing = [r for r in results 
                       if r["operation"] == target_operation 
                       and r["seed"] == seed 
                       and str(r["interp_range"]) == str(interp_range)]
            
            if not existing:
                test_configs.append({
                    'operation': target_operation,
                    'seed': seed,
                    'interp_range': interp_range,
                    'extrap_range': extrap_range,
                    'range_name': range_name
                })
    
    if not test_configs:
        print("  All tests already completed (cached results)")
        return
    
    print(f"  Running {len(test_configs)} tests concurrently...")
    
    # Split test configs into batches for workers
    batch_size = max(1, len(test_configs) // max_workers)
    test_batches = [test_configs[i:i + batch_size] for i in range(0, len(test_configs), batch_size)]
    
    completed_count = 0
    total_tests = len(test_configs)
    
    # Run tests concurrently
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit batches
        future_to_batch = {executor.submit(run_test_batch, batch): batch for batch in test_batches}
        
        # Process completed batches
        for future in as_completed(future_to_batch):
            try:
                batch_results = future.result()
                results.extend(batch_results)
                completed_count += len(batch_results)
                
                # Show progress
                progress_pct = (completed_count / total_tests) * 100
                print(f"  Progress: {completed_count}/{total_tests} ({progress_pct:.1f}%) - Latest batch: ", end="")
                
                # Show results from this batch
                for result in batch_results:
                    print("✅" if result["grokked"] else "❌", end="")
                    print(f"({result['seed']})", end=" ")
                print()
                
                # Save incrementally
                save_results(results)
                
            except Exception as exc:
                print(f'  Batch generated an exception: {exc}')
    
    # Show final summary for this operation
    print(f"\n  Final results for {target_operation.upper()}:")
    for interp_range, extrap_range, range_name in ranges_to_test:
        op_range_results = [r for r in results 
                           if r["operation"] == target_operation 
                           and str(r["interp_range"]) == str(interp_range)]
        
        if op_range_results:
            success_rate = sum(1 for r in op_range_results if r["grokked"]) / len(op_range_results) * 100
            print(f"    {range_name}: {success_rate:.0f}%")

def generate_final_table():
    """Generate and display the final results table."""
    results = load_existing_results()
    
    if not results:
        print("No results found!")
        return
        
    print(f"\n" + "=" * 70)
    print("RESULTS TABLE")
    print("=" * 70)
    
    # Create DataFrame for analysis
    data = []
    for interp_range, extrap_range, range_name in TEST_RANGES:
        for operation in OPERATIONS:
            op_range_results = [r for r in results 
                              if r["operation"] == operation and str(r["interp_range"]) == str(interp_range)]
            
            grokked_count = sum(1 for r in op_range_results if r["grokked"])
            total_count = len(op_range_results)
            success_rate = grokked_count / total_count * 100 if total_count > 0 else 0
            
            # Get average grok step for successful runs
            grok_steps = [r["grok_step"] for r in op_range_results if r["grok_step"] is not None]
            avg_grok_step = sum(grok_steps) / len(grok_steps) if grok_steps else None
            
            data.append({
                "Range": range_name,
                "Interp": str(interp_range),
                "Extrap": str(extrap_range),
                "Operation": operation.upper(),
                "Success_Rate": f"{success_rate:.0f}%",
                "Count": f"{grokked_count}/{total_count}",
                "Avg_Step": f"{avg_grok_step:.0f}" if avg_grok_step else "N/A"
            })
    
    # Print table
    df = pd.DataFrame(data)
    
    # Pivot table for better display
    pivot_success = df.pivot(index=["Range", "Interp", "Extrap"], columns="Operation", values="Success_Rate")
    pivot_steps = df.pivot(index=["Range", "Interp", "Extrap"], columns="Operation", values="Avg_Step")
    
    print("\nSUCCESS RATES (% passing across seeds):")
    print(pivot_success.to_string())
    
    print(f"\nAVERAGE GROK STEPS (for successful runs):")
    print(pivot_steps.to_string())
    
    # Overall summary
    total_grokked = sum(1 for r in results if r["grokked"])
    total_experiments = len(results)
    overall_success_rate = total_grokked / total_experiments * 100 if total_experiments > 0 else 0
    
    print(f"\nOVERALL SUMMARY:")
    print(f"Total experiments: {total_experiments}")
    print(f"Total successful: {total_grokked}")
    print(f"Overall success rate: {overall_success_rate:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run chunked comprehensive test')
    parser.add_argument('--operation', choices=OPERATIONS, help='Run specific operation only')
    parser.add_argument('--range-idx', type=int, help='Run specific range index only (0-8)')
    parser.add_argument('--table-only', action='store_true', help='Just generate table from existing results')
    parser.add_argument('--concurrent', action='store_true', help='Run tests concurrently (faster)')
    parser.add_argument('--workers', type=int, default=4, help='Number of concurrent workers (default: 4)')
    
    args = parser.parse_args()
    
    if args.table_only:
        generate_final_table()
    elif args.operation:
        if args.concurrent:
            run_operation_chunk_concurrent(args.operation, args.range_idx, args.workers)
        else:
            run_operation_chunk(args.operation, args.range_idx)
    else:
        print("CHUNKED COMPREHENSIVE FROZEN SELECTOR TABLE GENERATION")
        print("=" * 70)
        print("Use --operation [mul|add|sub|div] to run specific operation")
        print("Use --range-idx [0-8] with --operation to run specific range")
        print("Use --concurrent to run tests in parallel (much faster)")
        print("Use --workers N to specify number of concurrent workers (default: 4)")
        print("Use --table-only to generate table from existing results")
        print()
        print("SEQUENTIAL Examples:")
        print("  python chunked_comprehensive_test.py --operation mul")
        print("  python chunked_comprehensive_test.py --operation div --range-idx 0")
        print()
        print("CONCURRENT Examples (RECOMMENDED):")
        print("  python chunked_comprehensive_test.py --operation mul --concurrent")
        print("  python chunked_comprehensive_test.py --operation div --concurrent --workers 6")
        print("  python chunked_comprehensive_test.py --operation add --range-idx 2 --concurrent")
        print()
        print("TABLE Generation:")
        print("  python chunked_comprehensive_test.py --table-only")