from __future__ import annotations

import argparse
import concurrent.futures
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

# Keep this list in sync with run_multi_seed.py
RANGE_PAIRS: List[Tuple[List[float], List[float] | List[List[float]]]] = [
    ([-20.0, -10.0], [-40.0, -20.0]),
    ([-2.0, -1.0], [-6.0, -2.0]),
    ([-1.2, -1.1], [-6.1, -1.2]),
    ([-0.2, -0.1], [-2.0, -0.2]),
    ([-2.0, 2.0], [[-6.0, -2.0], [2.0, 6.0]]),
    ([0.1, 0.2], [0.2, 2.0]),
    ([1.0, 2.0], [2.0, 6.0]),
    ([1.1, 1.2], [1.2, 6.0]),
    ([10.0, 20.0], [20.0, 40.0]),
]


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

    def run_one(
        task: Tuple[int, List[float], List[float] | List[List[float]]],
    ) -> Tuple[int, str, int]:
        seed, inter_rng, extra_rng = task
        cmd: List[str] = [
            python_exec,
            "-u",
            str(script_path),
            "--layer-type",
            "DAG",
            "--operation",
            args.operation,
            "--input-size",
            str(args.input_size),
            "--batch-size",
            str(args.batch_size),
            "--max-iterations",
            str(args.max_iterations),
            "--learning-rate",
            str(args.learning_rate),
            "--log-interval",
            str(args.log_interval),
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
        label = f"{args.operation}-seed{seed}-inter{str(inter_rng).replace(' ', '')}-extra{str(extra_rng).replace(' ', '')}"
        env["WANDB_NAME"] = label
        print(f"[supervisor] starting {label}")
        proc = subprocess.run(cmd, check=False, env=env)
        return (proc.returncode, label, seed)

    if args.concurrency <= 1:
        for t in tasks:
            rc, label, _ = run_one(t)
            print(f"[supervisor] finished {label} rc={rc}")
    else:
        max_workers = max(1, int(args.concurrency))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(run_one, t) for t in tasks]
            for fut in concurrent.futures.as_completed(futures):
                try:
                    rc, label, _ = fut.result()
                    print(f"[supervisor] finished {label} rc={rc}")
                except Exception as exc:
                    print(f"[supervisor] task failed with exception: {exc}")


if __name__ == "__main__":
    main()
