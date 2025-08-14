from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from experiments.range_pairs import RANGE_PAIRS

# Fixed configuration (kept simple on purpose)
# OPS: List[str] = ["add", "sub", "mul", "div"]
OPS: List[str] = ["div"]
# Temporarily set this very low so we can see completion of each pod
SEEDS = 1
# SEEDS: int = 25
START_SEED: int = 0
INPUT_SIZE: int = 2
# BATCH_SIZE: int = 128
BATCH_SIZE: int = 1024  # This was working well for me locally
# Temporarily set this very low so we can see completion of each pod
# MAX_ITERATIONS: int = 1000
MAX_ITERATIONS: int = 50000
LEARNING_RATE: float = 1e-3
LOG_INTERVAL: int = 100


@dataclass
class RunResult:
    seed: int
    range_idx: int
    success: bool
    inter: float | None
    extra: float | None
    early_stopped: bool


def _parse_final_metrics(stdout_text: str) -> Tuple[float | None, float | None, bool]:
    inter_val: float | None = None
    extra_val: float | None = None
    early_stopped: bool = False
    for ln in stdout_text.splitlines():
        if "Early stopping at step" in ln:
            early_stopped = True
        if "loss_valid_inter:" in ln:
            # Extract the value after the colon
            parts = ln.split(":")
            if len(parts) < 2:
                raise ValueError(f"Malformed 'loss_valid_inter' line: {ln}")
            inter_val = float(parts[1].strip())
        if "loss_valid_extra:" in ln:
            # Extract the value after the colon
            parts = ln.split(":")
            if len(parts) < 2:
                raise ValueError(f"Malformed 'loss_valid_extra' line: {ln}")
            extra_val = float(parts[1].strip())
    return inter_val, extra_val, early_stopped


def run_single(
    repo_root: Path,
    op_name: str,
    seed: int,
    inter_rng,
    extra_rng,
    range_idx: int,
    max_iterations: int,
    batch_size: int,
    log_interval: int,
) -> RunResult:
    script_path = (
        repo_root / "nalm-benchmark" / "experiments" / "single_layer_benchmark.py"
    )
    python_exec = sys.executable
    args: List[str] = [
        python_exec,
        "-u",
        str(script_path),
        "--no-cuda",
        "--layer-type",
        "DAG",
        "--operation",
        op_name,
        "--input-size",
        str(INPUT_SIZE),
        "--batch-size",
        str(batch_size),
        "--max-iterations",
        str(max_iterations),
        "--learning-rate",
        str(LEARNING_RATE),
        "--log-interval",
        str(log_interval),
        "--clip-grad-norm",
        "1.0",
        "--interpolation-range",
        str(inter_rng),
        "--extrapolation-range",
        str(extra_rng),
        "--seed",
        str(seed),
    ]

    # Set up environment with proper Python path
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{repo_root}:{repo_root}/nalm-benchmark:{env.get('PYTHONPATH', '')}"
    )

    proc = subprocess.run(args, capture_output=True, text=True, env=env)
    stdout_text = proc.stdout or ""
    stderr_text = proc.stderr or ""

    # Parse both stdout and stderr since the benchmark might output to either
    inter, extra, early = _parse_final_metrics(stdout_text + "\n" + stderr_text)
    threshold = 1e-10
    success = (
        inter is not None
        and extra is not None
        and inter < threshold
        and extra < threshold
    )
    return RunResult(
        seed=seed,
        range_idx=range_idx,
        success=success,
        inter=inter,
        extra=extra,
        early_stopped=early,
    )


def _run_op_on_runpod(repo_root: Path, op_name: str) -> None:
    runpod_service_path = repo_root / "runpod_service" / "runpod_launcher.py"
    # Launch an on-pod supervisor that runs all seeds × ranges sequentially
    script_path = repo_root / "nalm-benchmark" / "experiments" / "op_supervisor.py"
    python_exec = sys.executable

    pod_name = f"multi-seed-DAG-{op_name}"

    base_args: List[str] = [
        "--operation",
        op_name,
        "--input-size",
        str(INPUT_SIZE),
        "--batch-size",
        str(BATCH_SIZE),
        "--max-iterations",
        str(MAX_ITERATIONS),
        "--learning-rate",
        str(LEARNING_RATE),
        "--log-interval",
        str(LOG_INTERVAL),
        "--clip-grad-norm",
        "1.0",
        "--start-seed",
        str(START_SEED),
        "--num-seeds",
        str(SEEDS),
        "--concurrency",
        "2",
    ]

    # Launch pod which will run all seeds × ranges for this op
    create_cmd: List[str] = [
        python_exec,
        "-u",
        str(runpod_service_path),
        str(script_path),
        *base_args,
        "--pod-name",
        pod_name,
        "--lifetime-minutes",
        "60",
    ]
    print(f"[runpod] Launching pod '{pod_name}' for op='{op_name}' (supervisor mode)")
    subprocess.run(create_cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multi-seed single-layer benchmark locally or on RunPod."
    )
    parser.add_argument(
        "--use-runpod",
        action="store_true",
        help="Launch one RunPod per operation and attach remaining seeds/ranges to that pod.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run locally with max iterations set to 2, batch size set to 2, and log interval set to 1 for a quick verification.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    effective_max_iters = 2 if args.test else MAX_ITERATIONS
    effective_batch_size = 2 if args.test else BATCH_SIZE
    effective_log_interval = 1 if args.test else LOG_INTERVAL

    # In test mode, always run locally
    if args.use_runpod and args.test:
        print(
            "[test] Ignoring --use-runpod because --test is set (forcing local run with 2 iterations, batch size 2, and log interval 1)."
        )

    if args.use_runpod and not args.test:
        # Launch one pod per op concurrently; don't stop the whole script if one fails
        import concurrent.futures

        print("[runpod] Launching pods for ops:", ", ".join(OPS))
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(OPS)) as ex:
            fut_to_op = {ex.submit(_run_op_on_runpod, repo_root, op): op for op in OPS}
            for fut in concurrent.futures.as_completed(fut_to_op):
                op = fut_to_op[fut]
                try:
                    fut.result()
                    print(f"[runpod] Pod launch submitted for op='{op}'")
                except Exception as exc:
                    print(f"[runpod] Failed to launch pod for op='{op}': {exc}")
        print("[runpod] All pod launch requests processed.")
        return

    # Local execution path
    summary: Dict[str, List[RunResult]] = {}
    for op in OPS:
        results: List[RunResult] = []
        for seed in range(START_SEED, START_SEED + SEEDS):
            for r_idx, (inter_rng, extra_rng) in enumerate(RANGE_PAIRS):
                res = run_single(
                    repo_root=repo_root,
                    op_name=op,
                    seed=seed,
                    inter_rng=inter_rng,
                    extra_rng=extra_rng,
                    range_idx=r_idx,
                    max_iterations=effective_max_iters,
                    batch_size=effective_batch_size,
                    log_interval=effective_log_interval,
                )
                status = "OK" if res.success else "FAIL"
                inter_s = f"{res.inter:.3e}" if res.inter is not None else "NA"
                extra_s = f"{res.extra:.3e}" if res.extra is not None else "NA"
                early_s = " early" if res.early_stopped else ""
                # Format range display - show actual values instead of just index
                inter_str = str(inter_rng).replace(" ", "")
                extra_str = str(extra_rng).replace(" ", "")
                print(
                    f"[{op}] seed={res.seed} inter={inter_str} extra={extra_str} {status} inter={inter_s} extra={extra_s}{early_s}"
                )
                results.append(res)
        summary[op] = results

    print("\nSummary:")
    for op, results in summary.items():
        total = len(results)
        passed = sum(1 for r in results if r.success)
        # Show seed and actual range values in summary
        seeds_ok = []
        for r in results:
            if r.success:
                inter_rng, extra_rng = RANGE_PAIRS[r.range_idx]
                inter_str = str(inter_rng).replace(" ", "")
                extra_str = str(extra_rng).replace(" ", "")
                seeds_ok.append(f"{r.seed}:{inter_str}/{extra_str}")
        print(
            f"- {op}: {passed}/{total} success (runs: {', '.join(seeds_ok) if seeds_ok else '-'})"
        )


if __name__ == "__main__":
    main()
