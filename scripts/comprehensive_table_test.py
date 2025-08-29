#!/usr/bin/env python3
"""
Comprehensive frozen selector test to generate percentage passing table.
"""

import argparse
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: pandas not available, table formatting will be limited")

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Warning: tqdm not available, progress bars disabled")

# Test configuration
TEST_SEEDS = [122, 223, 42, 777, 1337]
TEST_RANGES = [
    ([-2, 2], [[-6, -2], [2, 6]], "sym"),  # symmetric around 0
    ([-2, -1], [-6, -2], "neg"),  # negative moderate
    ([1, 2], [2, 6], "pos"),  # positive moderate
    ([-1.2, -1.1], [-6.1, -1.2], "n10"),  # negative narrow (around -1.1)
    ([0.1, 0.2], [0.2, 2], "p01"),  # positive small (0.1-0.2)
    ([-0.2, -0.1], [-2, -0.2], "n01"),  # negative small (-0.2 to -0.1)
    ([1.1, 1.2], [1.2, 6], "p11"),  # positive narrow (around 1.1)
    ([-20, -10], [-40, -20], "n20"),  # negative large (-20 to -10)
    ([10, 20], [20, 40], "p20"),  # positive large (10-20)
]

OPERATIONS = ["div", "sub", "mul", "add"]


def load_progress(progress_file):
    """Load existing progress from file."""
    if progress_file.exists():
        try:
            with open(progress_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {"completed": {}, "results": []}
    return {"completed": {}, "results": []}


def save_progress(progress_file, progress_data):
    """Save current progress to file."""
    progress_file.parent.mkdir(exist_ok=True)
    with open(progress_file, "w") as f:
        json.dump(progress_data, f, indent=2)


def get_experiment_key(seed, operation, range_name):
    """Get unique key for seed/operation/range combination."""
    return f"{seed}_{operation}_{range_name}"


def run_single_test(
    operation,
    seed,
    interp_range,
    extrap_range,
    range_name,
    show_progress=False,
    max_iterations=2000,
    restart_iter=None,
    disable_early_stopping=False,
    # batch_size=512,
    batch_size=128,
    disable_logging=False,
):
    """Run a single test and return result."""

    # Base command with updated hyperparameters
    cmd = [
        sys.executable,
        "experiments/single_layer_benchmark.py",
        "--layer-type",
        "DAG",
        "--no-open-browser",
        "--dag-depth",
        "1",
        "--operation",
        operation,
        "--seed",
        str(seed),
        "--input-size",
        "2",
        "--batch-size",
        str(batch_size),
        "--max-iterations",
        str(max_iterations),
        "--learning-rate",
        "1e-3",
        "--interpolation-range",
        str(interp_range),
        "--extrapolation-range",
        str(extrap_range),
        "--no-cuda",
        "--log-interval",
        (
            "99999" if disable_logging else "100"
        ),  # Very high interval when logging disabled
        "--clip-grad-norm",
        "0.01",
        "--note",
        f"seed{seed}_{range_name}",
    ]

    # Add restart argument if specified
    if restart_iter is not None:
        cmd.extend(["--restart-iter", str(restart_iter)])

    # Add disable early stopping argument if specified
    if disable_early_stopping:
        cmd.append("--disable-early-stopping")

    # Add disable logging argument if specified
    if disable_logging:
        cmd.append("--disable-logging")

    # Add frozen selector arguments based on operation
    if operation in ["mul", "add"]:
        cmd.append("--freeze-O-mul")
        frozen_config = "freeze_O_mul=True"
    elif operation in ["sub", "div"]:
        cmd.append("--freeze-O-div")
        frozen_config = "freeze_O_div=True"

    try:
        start_time = time.time()
        # Keep --no-open-browser to prevent browser spam, tqdm still works
        cmd_display = cmd

        # Use Popen for real-time tqdm display
        process = subprocess.Popen(
            cmd_display,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            universal_newlines=True,
            bufsize=1,
        )

        output_lines = []
        try:
            # Read output line by line in real-time
            for line in process.stdout:
                line_stripped = line.rstrip()
                if show_progress:
                    # Show training progress lines and important status messages
                    if (
                        re.match(r"^train \d+:", line_stripped)
                        or "Early stopping" in line_stripped
                        or "Early stopped" in line_stripped
                        or "finished:" in line_stripped
                    ):
                        print(line_stripped)
                # Default: suppress all subprocess output for cleaner display
                output_lines.append(line_stripped)

            # Wait for process to complete
            process.wait(timeout=120)
            duration = time.time() - start_time

        except subprocess.TimeoutExpired:
            process.kill()
            duration = 120
            raise

        grokked = False
        grok_step = None
        final_inter_loss = float("inf")
        final_sparsity_error = None
        nan_error = False

        # Check for NaN errors first
        for line in output_lines:
            if "nan" in line.lower() or "NaN" in line:
                nan_error = True
                break

        # Check for success based on evaluation mode
        if not nan_error:
            if disable_early_stopping:
                # Parse paper-faithful evaluation results
                for line in output_lines:
                    if "✓ PASSED" in line and "Dual threshold" in line:
                        grokked = True
                    elif "Success achieved at step:" in line:
                        try:
                            grok_step = int(line.split("step: ")[1])
                        except:
                            pass
                    elif "Final sparsity error:" in line:
                        try:
                            final_sparsity_error = float(line.split(":")[1].strip())
                        except:
                            pass
            else:
                # Check for early stopping (original logic)
                for line in output_lines:
                    if "Early stopping at step" in line:
                        grokked = True
                        try:
                            grok_step = int(line.split("step ")[1].split(":")[0])
                        except:
                            pass
                        # Extract sparsity from early stopping case
                        continue
                    elif "Final sparsity error:" in line and grokked:
                        try:
                            final_sparsity_error = float(line.split(":")[1].strip())
                        except:
                            pass

        # Extract final interpolation loss for fallback analysis
        if not grokked and not nan_error:
            for line in reversed(output_lines):
                if "- loss_valid_inter:" in line:
                    try:
                        final_inter_loss = float(line.split(":")[1].strip())
                        if math.isnan(final_inter_loss):
                            nan_error = True
                        break
                    except:
                        continue

        # Determine status
        if nan_error:
            status = "nan_error"
        elif grokked:
            status = "grokking"
        else:
            status = "not_grokking"

        return {
            "operation": operation,
            "seed": seed,
            "interp_range": interp_range,
            "extrap_range": extrap_range,
            "range_name": range_name,
            "frozen_config": frozen_config,
            "grokked": grokked,
            "nan_error": nan_error,
            "status": status,
            "grok_step": grok_step,
            "duration": duration,
            "final_inter_loss": final_inter_loss,
            "sparsity_error": final_sparsity_error,
        }

    except subprocess.TimeoutExpired:
        return {
            "operation": operation,
            "seed": seed,
            "interp_range": interp_range,
            "extrap_range": extrap_range,
            "range_name": range_name,
            "frozen_config": frozen_config,
            "grokked": False,
            "nan_error": False,
            "status": "not_grokking",
            "grok_step": None,
            "duration": 120,
            "final_inter_loss": float("inf"),
            "sparsity_error": None,
        }


def calculate_operation_statistics(results, operation):
    """Calculate statistics for a specific operation."""
    op_results = [r for r in results if r.get("operation") == operation]
    if not op_results:
        return None

    # Success rate statistics
    successes = sum(1 for r in op_results if r.get("status") == "grokking")
    trials = len(op_results)

    # Convergence time statistics (only for successful runs)
    convergence_times = [
        r.get("grok_step")
        for r in op_results
        if r.get("status") == "grokking" and r.get("grok_step") is not None
    ]

    # Sparsity error statistics (only for successful runs where available)
    sparsity_errors = [
        r.get("sparsity_error")
        for r in op_results
        if r.get("status") == "grokking" and r.get("sparsity_error") is not None
    ]

    return {
        "operation": operation,
        "success_rate": successes / trials if trials > 0 else 0.0,
        "successes": successes,
        "trials": trials,
        "convergence_times": convergence_times,
        "mean_convergence": (
            sum(convergence_times) / len(convergence_times)
            if convergence_times
            else float("inf")
        ),
        "sparsity_errors": sparsity_errors,
        "mean_sparsity": (
            sum(sparsity_errors) / len(sparsity_errors) if sparsity_errors else None
        ),
    }


def print_statistical_analysis(results):
    """Print statistical analysis with confidence intervals."""

    # Import confidence interval functions
    try:
        sys.path.append(str(Path(__file__).parent.parent / "misc"))
        from confidence_intervals import (binomial_confidence_interval,
                                          format_confidence_interval,
                                          gamma_confidence_interval)
    except ImportError as e:
        print(f"Warning: Could not import confidence interval functions: {e}")
        print("Skipping statistical analysis with confidence intervals")
        return

    print(f"\n{'='*60}")
    print("STATISTICAL ANALYSIS WITH CONFIDENCE INTERVALS")
    print(f"{'='*60}")

    # Overall statistics
    total_successes = sum(1 for r in results if r.get("status") == "grokking")
    total_trials = len(results)
    overall_success_rate = total_successes / total_trials if total_trials > 0 else 0.0

    # Overall success rate confidence interval
    overall_ci = binomial_confidence_interval(total_successes, total_trials)
    print(f"\nOVERALL SUCCESS RATE:")
    print(
        f"  Point estimate: {overall_success_rate:.3f} ({total_successes}/{total_trials})"
    )
    print(
        f"  95% Confidence Interval: {format_confidence_interval(overall_ci[0], overall_ci[1])}"
    )

    # Per-operation analysis
    print(f"\nPER-OPERATION ANALYSIS:")
    print(
        f"{'Operation':<10} {'Success Rate':<15} {'95% CI':<20} {'Mean Conv. Time':<18} {'95% CI':<20} {'Mean Sparsity':<15} {'95% CI':<20}"
    )
    print(f"{'-'*10} {'-'*15} {'-'*20} {'-'*18} {'-'*20} {'-'*15} {'-'*20}")

    for operation in OPERATIONS:
        stats = calculate_operation_statistics(results, operation)
        if stats is None:
            continue

        # Success rate CI
        success_ci = binomial_confidence_interval(stats["successes"], stats["trials"])
        success_rate_str = f"{stats['success_rate']:.3f}"
        success_ci_str = format_confidence_interval(success_ci[0], success_ci[1])

        # Convergence time CI (only if we have successful runs)
        if stats["convergence_times"]:
            conv_ci = gamma_confidence_interval(stats["convergence_times"])
            conv_mean_str = f"{stats['mean_convergence']:.1f}"
            conv_ci_str = format_confidence_interval(
                conv_ci[0], conv_ci[1], precision=1
            )
        else:
            conv_mean_str = "∞"
            conv_ci_str = "No successes"

        # Sparsity error CI (only if we have successful runs with sparsity data)
        if stats["sparsity_errors"]:
            from confidence_intervals import beta_confidence_interval

            sparsity_ci = beta_confidence_interval(stats["sparsity_errors"])
            sparsity_mean_str = (
                f"{stats['mean_sparsity']:.2e}"  # Scientific notation, 2 decimals
            )
            # Format CI bounds in scientific notation
            sparsity_ci_str = f"[{sparsity_ci[0]:.2e}, {sparsity_ci[1]:.2e}]"
        else:
            sparsity_mean_str = "—"
            sparsity_ci_str = "No data"

        print(
            f"{operation.upper():<10} {success_rate_str:<15} {success_ci_str:<20} {conv_mean_str:<18} {conv_ci_str:<20} {sparsity_mean_str:<15} {sparsity_ci_str:<20}"
        )

    # Range-specific analysis (if there are multiple ranges)
    range_names = set(r.get("range_name", "unknown") for r in results)
    if len(range_names) > 1:
        print(f"\nPER-RANGE ANALYSIS:")
        print(f"{'Range':<10} {'Success Rate':<15} {'95% CI':<25}")
        print(f"{'-'*10} {'-'*15} {'-'*25}")

        for range_name in sorted(range_names):
            range_results = [r for r in results if r.get("range_name") == range_name]
            range_successes = sum(
                1 for r in range_results if r.get("status") == "grokking"
            )
            range_trials = len(range_results)

            if range_trials > 0:
                range_success_rate = range_successes / range_trials
                range_ci = binomial_confidence_interval(range_successes, range_trials)

                success_rate_str = f"{range_success_rate:.3f}"
                ci_str = format_confidence_interval(range_ci[0], range_ci[1])

                print(f"{range_name:<10} {success_rate_str:<15} {ci_str:<25}")


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive frozen selector table test"
    )
    parser.add_argument(
        "--max-experiments",
        type=int,
        help="Limit total number of experiments (useful for debugging)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous progress if available",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Show training progress lines (train 0:, train 100:, etc.). Default is off for cleaner output.",
    )
    parser.add_argument(
        "--ops",
        type=str,
        nargs="+",
        choices=["mul", "add", "sub", "div"],
        help="Run experiments for only the specified operations (e.g., --ops div sub mul). If not specified, runs all operations.",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=25,
        help="Number of seeds to use, starting from 0 (default: 25, uses seeds 0-24)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Specific seeds to use (e.g., --seeds 0 42 123). Takes precedence over --num-seeds.",
    )
    parser.add_argument(
        "--ranges",
        type=str,
        nargs="+",
        choices=["sym", "neg", "pos", "n10", "p01", "n01", "p11", "n20", "p20"],
        help="Specific ranges to test (e.g., --ranges n01 pos sym). If not specified, tests all ranges.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3000,
        help="Maximum training iterations per experiment (default: 3000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for training (default: 512)",
    )
    parser.add_argument(
        "--disable-logging",
        action="store_true",
        default=True,
        help="Disable wandb logging and DAG taps to speed up runs (default: True, use --enable-logging to turn on)",
    )
    parser.add_argument(
        "--enable-logging",
        action="store_true",
        help="Enable wandb logging and DAG taps (overrides --disable-logging default)",
    )
    parser.add_argument(
        "--restart-iter",
        type=int,
        default=None,
        help="Restart training with re-initialized model if no early stop by this iteration",
    )
    parser.add_argument(
        "--disable-early-stopping",
        action="store_true",
        default=False,
        help="Disable early stopping and run for full iterations. Success determined post-hoc by checking dual threshold (paper-faithful evaluation).",
    )
    args = parser.parse_args()

    # Handle enable-logging override
    if args.enable_logging:
        args.disable_logging = False

    # Setup progress tracking
    output_dir = Path("experiment_results")
    output_dir.mkdir(exist_ok=True)
    progress_file = output_dir / "comprehensive_table_progress.json"

    # Load existing progress if resuming
    progress_data = (
        load_progress(progress_file)
        if args.resume
        else {"completed": {}, "results": []}
    )
    results = progress_data.get("results", [])

    print("COMPREHENSIVE FROZEN SELECTOR TABLE GENERATION")
    print("=" * 70)

    # Filter operations if --ops is specified
    if args.ops:
        operations_to_run = args.ops
        print(f"Using specified operations: {args.ops}")
    else:
        operations_to_run = OPERATIONS
        print(f"Using all operations: {OPERATIONS}")

    # Filter ranges if --ranges is specified
    if args.ranges:
        ranges_to_run = [
            (interp_range, extrap_range, range_name)
            for interp_range, extrap_range, range_name in TEST_RANGES
            if range_name in args.ranges
        ]
        print(f"Using specified ranges: {args.ranges}")
    else:
        ranges_to_run = TEST_RANGES
        print(f"Using all ranges: {[name for _, _, name in TEST_RANGES]}")

    # Generate seeds based on --seeds or --num-seeds argument
    if args.seeds:
        seeds_to_run = args.seeds
        print(f"Using explicit seeds: {seeds_to_run}")
    else:
        seeds_to_run = list(range(args.num_seeds))
        print(f"Using seeds 0-{args.num_seeds-1}: {seeds_to_run}")

    print(f"Operations: {operations_to_run}")
    print(f"Ranges: {len(ranges_to_run)} different interpolation/extrapolation pairs")
    print(
        f"Logging: {'DISABLED' if args.disable_logging else 'ENABLED'} ({'faster runs' if args.disable_logging else 'full wandb tracking'})"
    )

    total_experiments = len(seeds_to_run) * len(operations_to_run) * len(ranges_to_run)
    completed_count = len(progress_data.get("completed", {}))

    if args.max_experiments:
        total_experiments = min(total_experiments, args.max_experiments)
        print(f"Total experiments: {total_experiments} (LIMITED)")
    else:
        print(f"Total experiments: {total_experiments}")

    if args.resume and completed_count > 0:
        print(
            f"Resuming from previous run: {completed_count} experiments already completed"
        )
    print()

    completed = completed_count

    # Reordered loops: seed -> operation -> range
    for seed in seeds_to_run:
        print(f"\n🌱 Testing seed {seed}:")

        for operation in operations_to_run:
            print(f"  {operation.upper()}: ", end="")

            for interp_range, extrap_range, range_name in ranges_to_run:
                # Check if we've reached the experiment limit
                if args.max_experiments and completed >= args.max_experiments:
                    print(
                        f"\n⚠️ Reached experiment limit ({args.max_experiments}), stopping..."
                    )
                    break

                # Check if this experiment is already completed
                experiment_key = get_experiment_key(seed, operation, range_name)
                if experiment_key in progress_data["completed"]:
                    result = progress_data["completed"][experiment_key]
                    # Display cached result
                    if result["status"] == "grokking":
                        print("✅", end="")
                    elif result["status"] == "nan_error":
                        print("💥", end="")
                    else:
                        print("❌", end="")
                    print(f"({range_name[:3]})", end=" ")
                    continue

                # Run the experiment
                print(f"[{range_name[:3]}]", end="", flush=True)
                result = run_single_test(
                    operation,
                    seed,
                    interp_range,
                    extrap_range,
                    range_name,
                    args.show_progress,
                    args.max_iterations,
                    args.restart_iter,
                    args.disable_early_stopping,
                    args.batch_size,
                    args.disable_logging,
                )
                results.append(result)
                completed += 1

                # Save progress immediately
                progress_data["completed"][experiment_key] = result
                progress_data["results"] = results
                save_progress(progress_file, progress_data)

                # Display result
                print(
                    "\b" * (len(range_name[:3]) + 2), end=""
                )  # Clear the [xxx] indicator
                if result["status"] == "grokking":
                    print("✅", end="")
                elif result["status"] == "nan_error":
                    print("💥", end="")
                else:
                    print("❌", end="")
                print(f"({range_name[:3]})", end=" ")

            # Early break from ranges loop
            if args.max_experiments and completed >= args.max_experiments:
                break

            # Calculate success rate for this operation/seed combo
            # Collect all results for this operation/seed (from current run and cached)
            op_seed_results = []
            for key, cached_result in progress_data["completed"].items():
                cached_seed, cached_op, _ = key.split("_", 2)
                if int(cached_seed) == seed and cached_op == operation:
                    op_seed_results.append(cached_result)
            # Add any new results from current run (should be minimal overlap due to continue statement)
            for r in results:
                if r["operation"] == operation and r["seed"] == seed:
                    experiment_key = get_experiment_key(
                        seed,
                        operation,
                        next(
                            (
                                name
                                for ir, _, name in TEST_RANGES
                                if str(r["interp_range"]) == str(ir)
                            ),
                            "unknown",
                        ),
                    )
                    if experiment_key not in progress_data["completed"]:
                        op_seed_results.append(r)

            if op_seed_results:
                success_count = sum(
                    1 for r in op_seed_results if r["status"] == "grokking"
                )
                nan_count = sum(
                    1 for r in op_seed_results if r["status"] == "nan_error"
                )
                total_count = len(op_seed_results)
                success_rate = (
                    (success_count / total_count * 100) if total_count > 0 else 0
                )
                print(
                    f"→ {success_rate:.0f}% ({success_count}G/{nan_count}N/{total_count - success_count - nan_count}F)"
                )
            else:
                print("→ No results")

        # Early break from operations loop
        if args.max_experiments and completed >= args.max_experiments:
            break

        print(
            f"  Progress: {completed}/{total_experiments} ({completed/total_experiments*100:.1f}%)"
        )

        # Early break from seeds loop
        if args.max_experiments and completed >= args.max_experiments:
            break

    # Overall summary - include cached results (moved early for JSON saving)
    all_completed_results = list(progress_data.get("completed", {}).values())
    all_results = results + [r for r in all_completed_results if r not in results]

    # Generate comprehensive table
    print(f"\n" + "=" * 70)
    print("RESULTS TABLE")
    print("=" * 70)

    # Create DataFrame for analysis
    data = []
    for interp_range, extrap_range, range_name in ranges_to_run:
        for operation in operations_to_run:
            op_range_results = [
                r
                for r in results
                if r["operation"] == operation
                and str(r["interp_range"]) == str(interp_range)
            ]

            grokked_count = sum(1 for r in op_range_results if r["grokked"])
            total_count = len(op_range_results)
            success_rate = grokked_count / total_count * 100 if total_count > 0 else 0

            # Get average grok step for successful runs
            grok_steps = [
                r["grok_step"] for r in op_range_results if r["grok_step"] is not None
            ]
            avg_grok_step = sum(grok_steps) / len(grok_steps) if grok_steps else None

            data.append(
                {
                    "Range": range_name,
                    "Interp": str(interp_range),
                    "Extrap": str(extrap_range),
                    "Operation": operation.upper(),
                    "Success_Rate": f"{success_rate:.0f}%",
                    "Count": f"{grokked_count}/{total_count}",
                    "Avg_Step": f"{avg_grok_step:.0f}" if avg_grok_step else "N/A",
                }
            )

    # Print table
    if PANDAS_AVAILABLE:
        df = pd.DataFrame(data)

        # Pivot table for better display
        pivot_success = df.pivot(
            index=["Range", "Interp", "Extrap"],
            columns="Operation",
            values="Success_Rate",
        )
        pivot_steps = df.pivot(
            index=["Range", "Interp", "Extrap"], columns="Operation", values="Avg_Step"
        )

        print("\nSUCCESS RATES (% passing across seeds):")
        print(pivot_success.to_string())

        print(f"\nAVERAGE GROK STEPS (for successful runs):")
        print(pivot_steps.to_string())
    else:
        print("\nRESULTS TABLE (pandas not available, showing raw data):")
        for item in data:
            print(
                f"{item['Range']:15} {item['Operation']:4} {item['Success_Rate']:>6} {item['Count']:>8} {item['Avg_Step']:>8}"
            )

    # Save results
    timestamp = int(time.time())
    output_dir = Path("experiment_results")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"comprehensive_table_results_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "config": {
                    "seeds": seeds_to_run,
                    "ranges": [(ir, er, name) for ir, er, name in TEST_RANGES],
                    "operations": OPERATIONS,
                },
                "results": all_results,
                "summary_table": data,
            },
            f,
            indent=2,
        )

    print(f"\nDetailed results saved to: {output_file}")

    # Calculate final summary statistics
    total_grokked = sum(1 for r in all_results if r.get("status") == "grokking")
    total_nan = sum(1 for r in all_results if r.get("status") == "nan_error")
    total_failed = sum(1 for r in all_results if r.get("status") == "not_grokking")
    total_experiments_run = len(all_results)
    overall_success_rate = (
        (total_grokked / total_experiments_run * 100)
        if total_experiments_run > 0
        else 0
    )

    print(f"\nOVERALL SUMMARY:")
    print(f"Total experiments: {total_experiments_run}")
    print(f"Successful (grokking): {total_grokked}")
    print(f"NaN errors: {total_nan}")
    print(f"Failed (not grokking): {total_failed}")
    print(f"Overall success rate: {overall_success_rate:.1f}%")

    # Statistical analysis with confidence intervals
    print_statistical_analysis(all_results)

    # Clean up progress file on successful completion (only if we completed ALL experiments, not just limited ones)
    if not args.ops and len(progress_data.get("completed", {})) >= len(
        seeds_to_run
    ) * len(OPERATIONS) * len(TEST_RANGES):
        try:
            progress_file.unlink(missing_ok=True)
            print(f"Progress file cleaned up: {progress_file}")
        except:
            pass


if __name__ == "__main__":
    main()
