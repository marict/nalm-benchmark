#!/usr/bin/env python3
"""
Single frozen selector test - run one operation/seed/range combination.
"""

import argparse
import subprocess
import time
import json
import sys

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

def run_single_experiment(operation, seed, interp_range, extrap_range, 
                         linear_biased_init_G=None, freeze_O_selectors_div=None, 
                         freeze_O_selector_mul=None, dag_depth=None, use_dense_features=None):
    """Run a single experiment."""
    
    # Handle frozen selector configuration
    if freeze_O_selectors_div is None and freeze_O_selector_mul is None:
        # Use automatic selection based on operation (with DAG file modification as fallback)
        if operation in ["mul", "add"]:
            modify_dag_defaults(freeze_div=False, freeze_mul=True)
            frozen_config = "freeze_O_selector_mul=True (via DAG defaults)"
        elif operation in ["sub", "div"]:
            modify_dag_defaults(freeze_div=True, freeze_mul=False)
            frozen_config = "freeze_O_selectors_div=True (via DAG defaults)"
        else:
            print(f"Error: Unknown operation {operation}")
            return None
    else:
        # Use explicitly provided frozen selector arguments
        frozen_config = f"freeze_O_selectors_div={freeze_O_selectors_div}, freeze_O_selector_mul={freeze_O_selector_mul} (via args)"
    
    cmd = [
        "python", "experiments/single_layer_benchmark.py",
        "--layer-type", "DAG", "--no-open-browser",
        "--operation", operation, "--seed", str(seed),
        "--input-size", "2", "--batch-size", "512", 
        "--max-iterations", "2000", "--learning-rate", "1e-2",
        "--interpolation-range", str(interp_range),
        "--extrapolation-range", str(extrap_range),
        "--no-cuda", "--log-interval", "500"
    ]
    
    # Add DAG-specific arguments if provided
    if linear_biased_init_G is not None:
        cmd.extend(["--linear-biased-init-G", str(linear_biased_init_G).lower()])
    
    if freeze_O_selectors_div is not None:
        cmd.extend(["--freeze-O-selectors-div", str(freeze_O_selectors_div).lower()])
    
    if freeze_O_selector_mul is not None:
        cmd.extend(["--freeze-O-selector-mul", str(freeze_O_selector_mul).lower()])
    
    if dag_depth is not None:
        cmd.extend(["--dag-depth", str(dag_depth)])
    
    if use_dense_features is not None:
        cmd.extend(["--use-dense-features", str(use_dense_features).lower()])
    
    print(f"Testing {operation.upper()} seed {seed} range {interp_range} with {frozen_config}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=150)
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
        
        result_data = {
            "operation": operation,
            "seed": seed,
            "interp_range": interp_range,
            "extrap_range": extrap_range,
            "frozen_config": frozen_config,
            "success": result.returncode == 0,
            "grokked": grokked,
            "grok_step": grok_step,
            "duration": duration,
            "final_inter_loss": final_inter_loss,
        }
        
        if grokked:
            print(f"✅ GROKKED at step {grok_step} in {duration:.1f}s")
        else:
            print(f"❌ FAILED (loss: {final_inter_loss:.2e}) in {duration:.1f}s")
        
        return result_data
        
    except subprocess.TimeoutExpired:
        result_data = {
            "operation": operation,
            "seed": seed, 
            "interp_range": interp_range,
            "extrap_range": extrap_range,
            "frozen_config": frozen_config,
            "success": False,
            "grokked": False,
            "grok_step": None,
            "duration": 150,
            "final_inter_loss": float("inf"),
        }
        print(f"⏰ TIMEOUT after 150s")
        return result_data

def main():
    parser = argparse.ArgumentParser(description="Run single frozen selector test")
    parser.add_argument("operation", choices=["add", "sub", "mul", "div"], help="Arithmetic operation")
    parser.add_argument("seed", type=int, help="Random seed")
    parser.add_argument("--interp-range", default="[-2,2]", help="Interpolation range (default: [-2,2])")
    parser.add_argument("--extrap-range", default="[[-6,-2],[2,6]]", help="Extrapolation range (default: [[-6,-2],[2,6]])")
    parser.add_argument("--output", help="Output JSON file path")
    
    # DAG-specific parameters
    parser.add_argument("--linear-biased-init-G", type=lambda x: x.lower() in ['true', '1', 'yes'], 
                        help="Enable linear biased G initialization")
    parser.add_argument("--freeze-O-selectors-div", type=lambda x: x.lower() in ['true', '1', 'yes'],
                        help="Freeze O selectors for division pattern [1,-1,0,...]")
    parser.add_argument("--freeze-O-selector-mul", type=lambda x: x.lower() in ['true', '1', 'yes'],
                        help="Freeze O selectors for multiplication pattern [1,1,0,...]")
    parser.add_argument("--dag-depth", type=int, help="DAG depth")
    parser.add_argument("--use-dense-features", type=lambda x: x.lower() in ['true', '1', 'yes'],
                        help="Use dense features")
    
    args = parser.parse_args()
    
    # Parse ranges (simple eval for now - could be more robust)
    try:
        interp_range = eval(args.interp_range)
        extrap_range = eval(args.extrap_range)
    except:
        print(f"Error: Could not parse ranges {args.interp_range}, {args.extrap_range}")
        return 1
    
    # Fix argument names for compatibility
    linear_biased_init_G = getattr(args, 'linear_biased_init_G', None)
    freeze_O_selectors_div = getattr(args, 'freeze_O_selectors_div', None)  
    freeze_O_selector_mul = getattr(args, 'freeze_O_selector_mul', None)
    use_dense_features = getattr(args, 'use_dense_features', None)
    
    result = run_single_experiment(
        args.operation, args.seed, interp_range, extrap_range,
        linear_biased_init_G=linear_biased_init_G,
        freeze_O_selectors_div=freeze_O_selectors_div,
        freeze_O_selector_mul=freeze_O_selector_mul,
        dag_depth=args.dag_depth,
        use_dense_features=use_dense_features
    )
    
    if result is None:
        return 1
    
    # Save result if output file specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Result saved to {args.output}")
    
    # Return success/failure exit code
    return 0 if result["grokked"] else 1

if __name__ == "__main__":
    sys.exit(main())