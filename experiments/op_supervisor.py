from __future__ import annotations

import argparse
import concurrent.futures
import functools
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import runpod_service.wandb_setup as wandb

from experiments.range_pairs import RANGE_PAIRS


# Build a unique, readable W&B run label for a task
def build_label(operation: str, seed: int, inter_rng, extra_rng) -> str:
    inter_s = str(inter_rng).replace(" ", "")
    extra_s = str(extra_rng).replace(" ", "")
    return f"{operation}-seed{seed}-inter{inter_s}-extra{extra_s}"


def sanitize_label(s: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_.[],") else "_" for c in s)


def log_completion(
    operation: str,
    launched_count: int,
    completed_total: int,
    completed_ok: int,
    completed_failed: int,
) -> None:
    step_val = int(launched_count + completed_total)
    wandb.wrapper.log(
        {
            f"{operation}/completed_total": completed_total,
            f"{operation}/completed_ok": completed_ok,
            f"{operation}/completed_failed": completed_failed,
        },
        step=step_val,
        commit=True,
    )


def run_one(
    task: Tuple[int, List[float], List[float] | List[List[float]]],
    *,
    python_exec: str,
    script_path: Path,
    operation: str,
    input_size: int,
    batch_size: int,
    max_iterations: int,
    learning_rate: float,
    log_interval: int,
) -> Tuple[int, str, int]:
    seed, inter_rng, extra_rng = task
    cmd: List[str] = [
        python_exec,
        "-u",
        str(script_path),
        "--layer-type",
        "DAG",
        "--operation",
        operation,
        "--input-size",
        str(input_size),
        "--batch-size",
        str(batch_size),
        "--max-iterations",
        str(max_iterations),
        "--learning-rate",
        str(learning_rate),
        "--log-interval",
        str(log_interval),
        "--interpolation-range",
        str(inter_rng),
        "--extrapolation-range",
        str(extra_rng),
        "--seed",
        str(seed),
    ]
    env = os.environ.copy()
    env.setdefault("WANDB_PROJECT", "nalm-benchmark")
    # Ensure each subprocess creates a fresh W&B run (no resume)
    env.pop("WANDB_RUN_ID", None)
    env.pop("WANDB_RESUME", None)
    # Provide a unique, readable name
    label = build_label(operation, seed, inter_rng, extra_rng)
    env["WANDB_NAME"] = label

    # Verify that RUNPOD_POD_ID is set
    if "RUNPOD_POD_ID" not in env:
        raise RuntimeError("RUNPOD_POD_ID must be set in the environment")

    # Redirect child output to a dedicated log file on the network volume
    log_dir = Path("/runpod-volume") / "supervisor-logs" / operation
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{sanitize_label(label)}.log"
    print(f"Running {cmd} with env -> log: {log_path}")
    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        proc.wait()
    return (proc.returncode, f"{label} -> {log_path}", seed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="On-pod supervisor: runs all seeds × ranges for one op"
    )
    parser.add_argument("--operation", required=True)
    parser.add_argument("--input-size", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--max-iterations", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--log-interval", type=int, required=True)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=25)
    parser.add_argument("--concurrency", type=int, default=1)

    args = parser.parse_args()
    run = wandb.init_wandb()
    print(f"Initialized W&B, run id: {run.id}, url: {run.url}, name: {run.name}")

    # Resolve path to single_layer_benchmark.py robustly inside the pod
    here = Path(__file__).resolve()
    candidates = [
        # When repo root is the parent of 'nalm-benchmark'
        here.parents[2]
        / "nalm-benchmark"
        / "experiments"
        / "single_layer_benchmark.py",
        Path("/tmp/repo")
        / "nalm-benchmark"
        / "experiments"
        / "single_layer_benchmark.py",
        Path("/workspace")
        / "nalm-benchmark"
        / "experiments"
        / "single_layer_benchmark.py",
        Path.cwd() / "nalm-benchmark" / "experiments" / "single_layer_benchmark.py",
        # When repo root is the directory containing 'experiments'
        here.parent.parent / "experiments" / "single_layer_benchmark.py",
        Path("/tmp/repo") / "experiments" / "single_layer_benchmark.py",
        Path("/workspace") / "experiments" / "single_layer_benchmark.py",
        Path.cwd() / "experiments" / "single_layer_benchmark.py",
    ]
    script_path = next((p for p in candidates if p.exists()), None)
    if script_path is None:
        print(
            "[supervisor] ERROR: could not locate single_layer_benchmark.py. Checked:"
        )
        for p in candidates:
            print(f"  - {p}")
        sys.exit(2)
    python_exec = sys.executable

    tasks: List[Tuple[int, List[float], List[float] | List[List[float]]]] = []
    for seed in range(args.start_seed, args.start_seed + args.num_seeds):
        for inter_rng, extra_rng in RANGE_PAIRS:
            tasks.append((seed, inter_rng, extra_rng))

    run_one_bound = functools.partial(
        run_one,
        python_exec=python_exec,
        script_path=script_path,
        operation=args.operation,
        input_size=args.input_size,
        batch_size=args.batch_size,
        max_iterations=args.max_iterations,
        learning_rate=args.learning_rate,
        log_interval=args.log_interval,
    )

    total = len(tasks)
    launched = 0
    # Log initial points at step 0 to seed charts
    wandb.wrapper.log({f"{args.operation}/launched_total": 0}, step=0, commit=True)
    wandb.wrapper.log(
        {
            f"{args.operation}/completed_total": 0,
            f"{args.operation}/completed_ok": 0,
            f"{args.operation}/completed_failed": 0,
        },
        step=0,
        commit=True,
    )

    completed_total = 0
    completed_ok = 0
    completed_failed = 0
    if args.concurrency <= 1:
        for t in tasks:
            # Mark as launched at submission time
            seed, inter_rng, extra_rng = t
            launch_label = build_label(args.operation, seed, inter_rng, extra_rng)
            launched += 1
            print(f"[supervisor] launched {launch_label} ({launched}/{total})")
            # Log with an explicit step to ensure the series renders on W&B
            wandb.wrapper.log(
                {f"{args.operation}/launched_total": launched},
                step=launched,
                commit=True,
            )
            # Run in foreground (sequential)
            rc, msg, _ = run_one_bound(t)
            print(f"[supervisor] completed {msg} rc={rc}")
            completed_total += 1
            if rc == 0:
                completed_ok += 1
            else:
                completed_failed += 1
            log_completion(
                args.operation,
                launched,
                completed_total,
                completed_ok,
                completed_failed,
            )
    else:
        max_workers = max(1, int(args.concurrency))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures: list[concurrent.futures.Future] = []
            for t in tasks:
                seed, inter_rng, extra_rng = t
                launch_label = build_label(args.operation, seed, inter_rng, extra_rng)
                launched += 1
                print(f"[supervisor] launched {launch_label} ({launched}/{total})")
                wandb.wrapper.log(
                    {f"{args.operation}/launched_total": launched},
                    step=launched,
                    commit=True,
                )
                futures.append(ex.submit(run_one_bound, t))
            # Consume completions and log status
            for fut in concurrent.futures.as_completed(futures):
                try:
                    rc, msg, _seed = fut.result()
                    print(f"[supervisor] completed {msg} rc={rc}")
                    completed_total += 1
                    if rc == 0:
                        completed_ok += 1
                    else:
                        completed_failed += 1
                    log_completion(
                        args.operation,
                        launched,
                        completed_total,
                        completed_ok,
                        completed_failed,
                    )
                except Exception as exc:
                    print(f"[supervisor] task failed with exception: {exc}")
                    completed_total += 1
                    completed_failed += 1
                    log_completion(
                        args.operation,
                        launched,
                        completed_total,
                        completed_ok,
                        completed_failed,
                    )
            # Ensure all futures have completed
            for fut in futures:
                try:
                    fut.result()
                except Exception:
                    pass
    # Record a final summary metric for quick inspection
    wandb.wrapper.summary[f"{args.operation}/launched_total_final"] = launched
    wandb.wrapper.summary[f"{args.operation}/completed_total_final"] = completed_total
    wandb.wrapper.summary[f"{args.operation}/completed_ok_final"] = completed_ok
    wandb.wrapper.summary[f"{args.operation}/completed_failed_final"] = completed_failed
    wandb.wrapper.finish()


if __name__ == "__main__":
    main()
