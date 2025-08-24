#!/usr/bin/env python3
"""
Minimal frozen selector test - just verify one test case per operation group.
"""

import subprocess
import time

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

def run_test(operation, seed=122):
    """Run a single test."""
    cmd = [
        "python", "experiments/single_layer_benchmark.py",
        "--layer-type", "DAG", "--no-open-browser",
        "--operation", operation, "--seed", str(seed),
        "--input-size", "2", "--batch-size", "512", 
        "--max-iterations", "1500", "--learning-rate", "1e-2",
        "--interpolation-range", "[-2,2]",
        "--extrapolation-range", "[[-6,-2],[2,6]]",
        "--no-cuda", "--log-interval", "500"
    ]
    
    print(f"Testing {operation.upper()} with frozen selectors...")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=150)
        output_lines = result.stdout.split('\n') if result.stdout else []
        
        grokked = False
        grok_step = None
        
        # Check for early stopping
        for line in output_lines:
            if "Early stopping at step" in line:
                grokked = True
                try:
                    grok_step = int(line.split("step ")[1].split(":")[0])
                except:
                    pass
                break
        
        if grokked:
            print(f"✅ {operation.upper()}: GROKKED at step {grok_step}")
            return True
        else:
            # Check final loss
            for line in reversed(output_lines):
                if "- loss_valid_inter:" in line:
                    try:
                        final_loss = float(line.split(":")[1].strip())
                        print(f"❌ {operation.upper()}: Failed (loss: {final_loss:.2e})")
                        return False
                    except:
                        continue
            print(f"❌ {operation.upper()}: Failed (unknown)")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {operation.upper()}: Timeout")
        return False

def main():
    print("MINIMAL FROZEN SELECTOR TEST")
    print("=" * 50)
    
    results = {}
    
    # Test MUL with frozen multiplication selectors
    print("\n🔸 Testing MUL with freeze_O_selector_mul=True")
    modify_dag_defaults(freeze_div=False, freeze_mul=True)
    results['mul'] = run_test('mul')
    
    # Test ADD with frozen multiplication selectors  
    print("\n🔸 Testing ADD with freeze_O_selector_mul=True")
    # No need to change DAG again, still has freeze_mul=True
    results['add'] = run_test('add')
    
    # Test DIV with frozen division selectors
    print("\n🔸 Testing DIV with freeze_O_selectors_div=True")
    modify_dag_defaults(freeze_div=True, freeze_mul=False)
    results['div'] = run_test('div')
    
    # Test SUB with frozen division selectors
    print("\n🔸 Testing SUB with freeze_O_selectors_div=True")
    # No need to change DAG again, still has freeze_div=True
    results['sub'] = run_test('sub')
    
    # Summary
    print(f"\n" + "=" * 50)
    print("MINIMAL TEST SUMMARY")
    print("=" * 50)
    
    total_success = sum(results.values())
    total_tests = len(results)
    
    print(f"MUL (frozen mul selectors): {'✅' if results['mul'] else '❌'}")
    print(f"ADD (frozen mul selectors): {'✅' if results['add'] else '❌'}")
    print(f"DIV (frozen div selectors): {'✅' if results['div'] else '❌'}")
    print(f"SUB (frozen div selectors): {'✅' if results['sub'] else '❌'}")
    print(f"Overall: {total_success}/{total_tests} operations successful")

if __name__ == "__main__":
    main()