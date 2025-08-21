#!/usr/bin/env python3
"""
Multi-seed experiment runner for testing grokking success rates.

This script runs multiple experiments with different seeds and reports
the average grokking success rate. A grok is defined as early stopping
due to both interpolation and extrapolation errors going to near-zero.

Can generate NALU-style benchmark tables comparing success rates, convergence
iterations, and sparsity errors across different interpolation ranges.
"""

import argparse
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ExperimentResult:
    """Results from a single experiment."""

    seed: int
    grokked: bool
    early_stop_step: Optional[int]
    sparsity_error: Optional[float]
    final_interpolation_error: Optional[float]
    final_extrapolation_error: Optional[float]
    status: str
    duration: float


@dataclass
class RangeResults:
    """Aggregated results for a specific range."""

    range_str: str
    success_rate: float
    success_rate_ci: Tuple[float, float]  # 95% confidence interval
    solved_at_mean: Optional[float]
    solved_at_ci: Optional[Tuple[float, float]]
    sparsity_error_mean: Optional[float]
    sparsity_error_ci: Optional[Tuple[float, float]]
    individual_results: List[ExperimentResult]


def parse_early_stop_from_output(output: str) -> Tuple[bool, Optional[int]]:
    """Parse whether the experiment early stopped and at what iteration."""
    lines = output.split("\n")

    # Look for early stopping message
    for line in lines:
        if "Early stopping at step" in line:
            try:
                # Extract step number from "Early stopping at step 1234: ..."
                step_part = line.split("Early stopping at step ")[1]
                step_num = int(step_part.split(":")[0])
                return True, step_num
            except (IndexError, ValueError):
                return True, None

    return False, None


def parse_sparsity_error_from_output(output: str) -> Optional[float]:
    """Extract sparsity error from the output logs."""
    lines = output.split("\n")

    # Look for regularization or sparsity-related metrics
    # Common patterns: "sparsity:", "l1:", "reg:", etc.
    sparsity_patterns = [
        r"sparsity[:\s]+([0-9.e-]+)",
        r"l1[:\s]+([0-9.e-]+)",
        r"regularization[:\s]+([0-9.e-]+)",
        r"reg[:\s]+([0-9.e-]+)",
    ]

    for line in lines:
        for pattern in sparsity_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue

    return None


def parse_final_errors_from_output(
    output: str,
) -> Tuple[Optional[float], Optional[float]]:
    """Extract final interpolation and extrapolation errors."""
    lines = output.split("\n")

    interp_error = None
    extrap_error = None

    # Look for final error reporting patterns
    for line in lines:
        # Early stopping line: "Early stopping at step 1234: inter=1.234e-10, extra=5.678e-11"
        early_stop_match = re.search(
            r"Early stopping.*inter=([0-9.e-]+).*extra=([0-9.e-]+)", line
        )
        if early_stop_match:
            try:
                interp_error = float(early_stop_match.group(1))
                extrap_error = float(early_stop_match.group(2))
            except ValueError:
                pass

        # Training log patterns
        interp_match = re.search(r"train.*?([0-9.e-]+).*?inter.*?([0-9.e-]+)", line)
        if interp_match:
            try:
                interp_error = float(interp_match.group(2))
            except ValueError:
                pass

        extrap_match = re.search(r"extra.*?([0-9.e-]+)", line)
        if extrap_match:
            try:
                extrap_error = float(extrap_match.group(1))
            except ValueError:
                pass

    return interp_error, extrap_error


def run_single_seed_experiment(
    base_args: List[str], seed: int, experiment_id: int, total_experiments: int
) -> ExperimentResult:
    """Run a single experiment with the given seed."""
    # Create command with the new seed
    cmd = base_args + ["--seed", str(seed)]

    # Add experiment-specific note
    note_updated = False
    for i, arg in enumerate(cmd):
        if arg == "--note" and i + 1 < len(cmd):
            cmd[i + 1] = f"{cmd[i + 1]}_seed{seed}"
            note_updated = True
            break

    if not note_updated:
        cmd.extend(["--note", f"multiseed_seed{seed}"])

    print(f"[Experiment {experiment_id}/{total_experiments}] Running seed {seed}...")
    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout per experiment
        )

        duration = time.time() - start_time

        if result.returncode == 0:
            # Parse detailed results
            grokked, early_stop_step = parse_early_stop_from_output(result.stdout)
            sparsity_error = parse_sparsity_error_from_output(result.stdout)
            final_interp_error, final_extrap_error = parse_final_errors_from_output(
                result.stdout
            )

            if grokked:
                step_info = f" at step {early_stop_step}" if early_stop_step else ""
                status = f"✅ GROKKED{step_info} ({duration:.1f}s)"
            else:
                status = f"❌ No grok ({duration:.1f}s)"

            return ExperimentResult(
                seed=seed,
                grokked=grokked,
                early_stop_step=early_stop_step,
                sparsity_error=sparsity_error,
                final_interpolation_error=final_interp_error,
                final_extrapolation_error=final_extrap_error,
                status=status,
                duration=duration,
            )
        else:
            status = f"❌ ERROR (code {result.returncode}, {duration:.1f}s)"
            print(f"STDERR: {result.stderr}")
            return ExperimentResult(
                seed=seed,
                grokked=False,
                early_stop_step=None,
                sparsity_error=None,
                final_interpolation_error=None,
                final_extrapolation_error=None,
                status=status,
                duration=duration,
            )

    except subprocess.TimeoutExpired:
        status = f"❌ TIMEOUT (>600s)"
        return ExperimentResult(
            seed=seed,
            grokked=False,
            early_stop_step=None,
            sparsity_error=None,
            final_interpolation_error=None,
            final_extrapolation_error=None,
            status=status,
            duration=600.0,
        )
    except Exception as e:
        status = f"❌ EXCEPTION: {str(e)}"
        return ExperimentResult(
            seed=seed,
            grokked=False,
            early_stop_step=None,
            sparsity_error=None,
            final_interpolation_error=None,
            final_extrapolation_error=None,
            status=status,
            duration=time.time() - start_time,
        )


def calculate_confidence_interval(
    values: List[float], confidence: float = 0.95
) -> Tuple[float, float]:
    """Calculate confidence interval for a list of values."""
    if not values:
        return (0.0, 0.0)

    values_array = np.array(values)
    mean = np.mean(values_array)

    if len(values) == 1:
        return (mean, mean)

    # Use t-distribution for small samples
    from scipy import stats

    try:
        sem = stats.sem(values_array)  # Standard error of mean
        ci = stats.t.interval(confidence, len(values) - 1, loc=mean, scale=sem)
        return ci
    except ImportError:
        # Fallback to normal approximation if scipy not available
        std = np.std(values_array, ddof=1)
        margin = 1.96 * std / np.sqrt(len(values))  # 95% CI approximation
        return (mean - margin, mean + margin)


def calculate_binomial_confidence_interval(
    successes: int, total: int, confidence: float = 0.95
) -> Tuple[float, float]:
    """Calculate confidence interval for binomial proportion (success rate)."""
    if total == 0:
        return (0.0, 0.0)

    p = successes / total

    if total == 1:
        return (p, p)

    # Wilson score interval (better than normal approximation for small n)
    z = 1.96  # 95% confidence
    term1 = p + z**2 / (2 * total)
    term2 = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2))
    denominator = 1 + z**2 / total

    ci_lower = (term1 - term2) / denominator
    ci_upper = (term1 + term2) / denominator

    return (max(0.0, ci_lower), min(1.0, ci_upper))


def aggregate_results(results: List[ExperimentResult], range_str: str) -> RangeResults:
    """Aggregate individual experiment results into summary statistics."""
    total_experiments = len(results)
    successful_results = [r for r in results if r.grokked]
    success_count = len(successful_results)

    # Success rate and CI
    success_rate = success_count / total_experiments if total_experiments > 0 else 0.0
    success_rate_ci = calculate_binomial_confidence_interval(
        success_count, total_experiments
    )

    # Solved at (convergence steps) - only for successful experiments
    solved_at_mean = None
    solved_at_ci = None
    if successful_results:
        convergence_steps = [
            r.early_stop_step
            for r in successful_results
            if r.early_stop_step is not None
        ]
        if convergence_steps:
            solved_at_mean = np.mean(convergence_steps)
            solved_at_ci = calculate_confidence_interval(convergence_steps)

    # Sparsity error - from all experiments that completed successfully
    sparsity_error_mean = None
    sparsity_error_ci = None
    completed_results = [r for r in results if r.sparsity_error is not None]
    if completed_results:
        sparsity_errors = [r.sparsity_error for r in completed_results]
        sparsity_error_mean = np.mean(sparsity_errors)
        sparsity_error_ci = calculate_confidence_interval(sparsity_errors)

    return RangeResults(
        range_str=range_str,
        success_rate=success_rate,
        success_rate_ci=success_rate_ci,
        solved_at_mean=solved_at_mean,
        solved_at_ci=solved_at_ci,
        sparsity_error_mean=sparsity_error_mean,
        sparsity_error_ci=sparsity_error_ci,
        individual_results=results,
    )


def format_nalu_table(operation: str, range_results: List[RangeResults]) -> str:
    """Format results in NALU benchmark table style."""
    lines = []
    lines.append(f"\nTable: Results for {operation}")
    lines.append(
        "Comparison of success-rate, model convergence iteration, and sparsity error"
    )
    lines.append(
        "with 95% confidence interval. Each value is a summary of 25 different seeds."
    )
    lines.append("")
    lines.append(
        "Model        Range           Success       Solved at        Sparsity error"
    )
    lines.append("                             Rate          Mean             Mean")
    lines.append("-" * 80)

    for range_result in range_results:
        # Success rate with CI (match NALU format: +18%/-19%)
        success_pct = f"{range_result.success_rate:.0%}"
        if range_result.success_rate_ci:
            ci_upper_margin = (
                range_result.success_rate_ci[1] - range_result.success_rate
            )
            ci_lower_margin = (
                range_result.success_rate - range_result.success_rate_ci[0]
            )
            success_str = f"{success_pct} +{ci_upper_margin:.0%}/-{ci_lower_margin:.0%}"
        else:
            success_str = success_pct

        # Solved at (convergence steps)
        if range_result.solved_at_mean is not None:
            solved_mean = f"{range_result.solved_at_mean:.1f}·10³"
            if range_result.solved_at_ci:
                ci_range = range_result.solved_at_ci[1] - range_result.solved_at_ci[0]
                solved_str = f"{solved_mean} ±{ci_range/2:.1f}·10³"
            else:
                solved_str = solved_mean
        else:
            solved_str = "—"

        # Sparsity error
        if range_result.sparsity_error_mean is not None:
            sparsity_mean = f"{range_result.sparsity_error_mean:.1e}"
            if range_result.sparsity_error_ci:
                ci_range = (
                    range_result.sparsity_error_ci[1]
                    - range_result.sparsity_error_ci[0]
                )
                sparsity_str = f"{sparsity_mean} ±{ci_range/2:.1e}"
            else:
                sparsity_str = sparsity_mean
        else:
            sparsity_str = "—"

        lines.append(
            f"DAG          {range_result.range_str:<15} {success_str:<13} {solved_str:<16} {sparsity_str}"
        )

    return "\n".join(lines)


def run_multiseed_experiment(base_args: List[str], seed_start: int, seed_end: int):
    """Run experiments across a range of seeds and report statistics."""
    seeds = list(range(seed_start, seed_end + 1))
    total_experiments = len(seeds)

    print(
        f"🧪 Running multi-seed experiment with seeds {seed_start} to {seed_end} ({total_experiments} total)"
    )
    print("=" * 80)

    results = []
    start_time = time.time()

    for i, seed in enumerate(seeds, 1):
        result = run_single_seed_experiment(base_args, seed, i, total_experiments)
        results.append(result)
        print(f"  {result.status}")

    total_duration = time.time() - start_time

    # Calculate statistics
    grokked_count = sum(1 for r in results if r.grokked)
    success_rate = grokked_count / total_experiments

    # Report summary
    print("\n" + "=" * 80)
    print("🏁 MULTI-SEED EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"Total experiments: {total_experiments}")
    print(f"Grokked: {grokked_count}")
    print(f"Failed: {total_experiments - grokked_count}")
    print(f"Success rate: {success_rate:.2%}")
    print(f"Total time: {total_duration:.1f}s")

    successful_results = [
        r for r in results if r.grokked and r.early_stop_step is not None
    ]
    if successful_results:
        grok_steps = [r.early_stop_step for r in successful_results]
        avg_grok_step = np.mean(grok_steps)
        print(f"Average grok step: {avg_grok_step:.1f}")
        if len(grok_steps) > 1:
            min_grok = min(grok_steps)
            max_grok = max(grok_steps)
            print(f"Grok step range: {min_grok} - {max_grok}")

    print(f"\nDetailed results:")
    for result in results:
        step_info = (
            f" (step {result.early_stop_step})" if result.early_stop_step else ""
        )
        grok_status = "GROKKED" if result.grokked else "FAILED"
        print(f"  Seed {result.seed:2d}: {grok_status}{step_info}")

    return results


def run_benchmark_across_ranges(
    base_args: List[str],
    seed_start: int,
    seed_end: int,
    ranges: List[str],
    operation: str,
) -> None:
    """Run benchmark across multiple interpolation ranges and generate NALU-style table."""

    # Define NALU paper range mappings: interpolation -> extrapolation
    # Using Python list format for ast.literal_eval compatibility
    nalu_range_mappings = {
        "[-20, -10]": "[-40, -20]",
        "[-2, -1]": "[-6, -2]",
        "[-1.2, -1.1]": "[-6.1, -1.2]",
        "[-0.2, -0.1]": "[-2, -0.2]",
        "[-2, 2]": "[[-6, -2], [2, 6]]",
        "[0.1, 0.2]": "[0.2, 2]",
        "[1, 2]": "[2, 6]",
        "[1.1, 1.2]": "[1.2, 6]",
        "[10, 20]": "[20, 40]",
    }

    all_range_results = []

    for range_str in ranges:
        print(f"\n{'='*80}")
        print(f"🎯 Testing range {range_str}")
        print(f"{'='*80}")

        # Get corresponding extrapolation range
        extrap_range = nalu_range_mappings.get(range_str)
        if extrap_range is None:
            print(
                f"⚠️  Warning: No standard extrapolation range for {range_str}, using default"
            )

        # Update both interpolation and extrapolation ranges in base args
        range_args = base_args.copy()

        # Find and replace interpolation-range argument
        for i, arg in enumerate(range_args):
            if arg == "--interpolation-range" and i + 1 < len(range_args):
                range_args[i + 1] = range_str
                break
        else:
            # Add interpolation range if not present
            range_args.extend(["--interpolation-range", range_str])

        # Find and replace extrapolation-range argument
        if extrap_range:
            for i, arg in enumerate(range_args):
                if arg == "--extrapolation-range" and i + 1 < len(range_args):
                    range_args[i + 1] = extrap_range
                    break
            else:
                # Add extrapolation range if not present
                range_args.extend(["--extrapolation-range", extrap_range])

        print(f"   Interpolation: {range_str}")
        print(f"   Extrapolation: {extrap_range if extrap_range else 'default'}")

        # Run experiments for this range
        results = run_multiseed_experiment(range_args, seed_start, seed_end)

        # Aggregate results
        range_result = aggregate_results(results, range_str)
        all_range_results.append(range_result)

    # Generate and print NALU-style table
    table = format_nalu_table(operation, all_range_results)
    print("\n" + "=" * 80)
    print("📊 NALU-STYLE BENCHMARK TABLE")
    print("=" * 80)
    print(table)


def get_standard_nalu_ranges() -> List[str]:
    """Get the standard NALU benchmark interpolation ranges."""
    return [
        "[-20, -10]",
        "[-2, -1]",
        "[-1.2, -1.1]",
        "[-0.2, -0.1]",
        "[-2, 2]",
        "[0.1, 0.2]",
        "[1, 2]",
        "[1.1, 1.2]",
        "[10, 20]",
    ]


def main():
    """Main function for multi-seed experiment runner."""
    parser = argparse.ArgumentParser(
        description="Run multi-seed experiments to test grokking success rates"
    )

    parser.add_argument(
        "--seed-range",
        required=True,
        type=str,
        help="Range of seeds to test, e.g., '[0,9]' for seeds 0-9",
    )

    parser.add_argument(
        "--ranges",
        type=str,
        help="List of interpolation ranges to test, e.g., \"['[-0.2,-0.1]', '[-1.2,-1.1]']\". Uses standard NALU paper extrapolation ranges. If provided, generates NALU-style benchmark table.",
    )

    parser.add_argument(
        "--operation",
        type=str,
        help="Operation being tested (for table formatting), e.g., 'addition', 'multiplication'",
    )

    parser.add_argument(
        "--base-cmd",
        required=True,
        nargs=argparse.REMAINDER,
        help="Base command arguments to pass to single_layer_benchmark.py",
    )

    args = parser.parse_args()

    # Parse seed range
    try:
        import ast

        seed_range = ast.literal_eval(args.seed_range)
        if not isinstance(seed_range, list) or len(seed_range) != 2:
            raise ValueError("seed-range must be a list of exactly 2 values")
        seed_start, seed_end = seed_range
    except Exception as e:
        print(f"❌ Error parsing seed-range: {e}")
        print("Example usage: --seed-range '[0,9]'")
        sys.exit(1)

    # Build base command
    base_cmd = [sys.executable, "experiments/single_layer_benchmark.py"] + args.base_cmd

    # Ensure --no-open-browser is set for automated runs
    if "--no-open-browser" not in base_cmd:
        base_cmd.append("--no-open-browser")

    print(f"Base command: {' '.join(base_cmd)}")
    print(f"Testing seeds: {seed_start} to {seed_end}")

    # Check if we're running benchmark across multiple ranges
    if args.ranges:
        try:
            ranges = ast.literal_eval(args.ranges)
            if not isinstance(ranges, list):
                raise ValueError("ranges must be a list")

            operation = args.operation if args.operation else "unknown"
            print(f"Benchmark mode: Testing {len(ranges)} ranges for {operation}")
            print()

            # Run benchmark across ranges
            run_benchmark_across_ranges(
                base_cmd, seed_start, seed_end, ranges, operation
            )
        except Exception as e:
            print(f"❌ Error parsing ranges: {e}")
            print(
                "Example usage: --ranges \"['[-2,-1]', '[1,2]']\" --operation addition"
            )
            sys.exit(1)
    else:
        # Run single range experiment
        print()
        run_multiseed_experiment(base_cmd, seed_start, seed_end)


if __name__ == "__main__":
    main()
