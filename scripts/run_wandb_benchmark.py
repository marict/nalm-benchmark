#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

from experiments import \
    wandb_setup as wandb  # exposes wandb.run and wandb.wrapper

TRAIN_LINE_RE = re.compile(
    r"^train\s+(\d+):\s+([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?),\s+inter:\s+([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?),\s+extra:\s+([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NALM single_layer_benchmark.py and stream metrics to wandb by parsing stdout."
    )
    parser.add_argument("--name", default=None, help="wandb run name (optional)")
    parser.add_argument("--notes", default=None, help="wandb notes (optional)")
    parser.add_argument(
        "--cwd",
        default=str(Path(__file__).resolve().parents[1]),
        help="Working directory to run benchmark from (defaults to repo root)",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use",
    )
    parser.add_argument(
        "benchmark_args",
        nargs=argparse.REMAINDER,
        help="Arguments to pass to experiments/single_layer_benchmark.py; prefix with --",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Ensure benchmark args present
    bench_argv = args.benchmark_args
    if bench_argv and bench_argv[0] == "--":
        bench_argv = bench_argv[1:]
    if not bench_argv:
        print(
            "ERROR: provide benchmark args after --, e.g. -- --no-cuda --layer-type DAG --operation add",
            file=sys.stderr,
        )
        return 2

    # W&B already initialized by import side-effect; optionally set presentation fields
    if args.name:
        try:
            wandb.run.name = args.name
        except Exception:
            pass
    if args.notes:
        try:
            wandb.run.summary["notes"] = args.notes
        except Exception:
            pass

    # Build command
    cwd = Path(args.cwd)
    script = cwd / "experiments" / "single_layer_benchmark.py"
    cmd = [args.python, str(script)] + bench_argv

    env = os.environ.copy()

    print("Launching:")
    print(" ", shlex.join(cmd))
    print(" cwd=", cwd)

    # Stream stdout, parse metrics
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            line = line.rstrip("\n")
            print(line)
            m = TRAIN_LINE_RE.match(line)
            if m:
                step = int(m.group(1))
                train_loss = float(m.group(2))
                inter_mse = float(m.group(3))
                extra_mse = float(m.group(4))
                wandb.wrapper.log(
                    {
                        "train/mse": train_loss,
                        "valid/inter_mse": inter_mse,
                        "test/extra_mse": extra_mse,
                    },
                    step=step,
                )
    finally:
        proc.wait()
        code = proc.returncode
        # Mark run status
        if code == 0:
            wandb.run.summary["status"] = "completed"
        else:
            wandb.run.summary["status"] = f"failed:{code}"
        wandb.wrapper.finish()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
