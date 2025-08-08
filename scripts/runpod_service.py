#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


def env_flag(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except Exception:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RunPod service wrapper for NALM benchmark with W&B logging")
    parser.add_argument("--run-name", dest="run_name", default=None, help="W&B run name")
    parser.add_argument("--notes", default=None, help="W&B notes")
    parser.add_argument("--wandb-api-key", dest="wandb_api_key", default=None, help="W&B API key (optional, else env)")
    parser.add_argument("--layer-type", dest="layer_type", default=None, help="Layer type, e.g. DAG")
    parser.add_argument("--operation", dest="operation", default=None, help="Operation: add|sub|mul|div")
    parser.add_argument("--input-size", dest="input_size", type=int, default=None)
    parser.add_argument("--hidden-size", dest="hidden_size", type=int, default=None)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    parser.add_argument("--max-iterations", dest="max_iterations", type=int, default=None)
    parser.add_argument("--log-interval", dest="log_interval", type=int, default=None)
    parser.add_argument("--learning-rate", dest="learning_rate", type=float, default=None)
    parser.add_argument("--clip-grad-norm", dest="clip_grad_norm", default=None)
    parser.add_argument("--no-cuda", dest="no_cuda", action="store_true", help="Force CPU")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # Required: WANDB_API_KEY should be supplied via pod env
    if args.wandb_api_key:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
    if not os.getenv("WANDB_API_KEY"):
        print("ERROR: WANDB_API_KEY not set in environment (or via --wandb-api-key).", file=sys.stderr)
        return 2

    # Benchmark knobs from environment (reasonable defaults)
    layer_type = args.layer_type or os.getenv("LAYER_TYPE", "DAG")
    operation = args.operation or os.getenv("OPERATION", "mul")
    input_size = args.input_size if args.input_size is not None else env_int("INPUT_SIZE", 4)
    hidden_size = args.hidden_size if args.hidden_size is not None else env_int("HIDDEN_SIZE", 2)
    batch_size = args.batch_size if args.batch_size is not None else env_int("BATCH_SIZE", 1000)
    max_iterations = args.max_iterations if args.max_iterations is not None else env_int("MAX_ITERATIONS", 300000)
    log_interval = args.log_interval if args.log_interval is not None else env_int("LOG_INTERVAL", 1000)
    learning_rate = args.learning_rate if args.learning_rate is not None else env_float("LEARNING_RATE", 1e-4)
    clip_grad_norm = args.clip_grad_norm or os.getenv("CLIP_GRAD_NORM", "1.0")
    no_cuda = args.no_cuda if args.no_cuda else env_flag("NO_CUDA", True)

    run_name = args.run_name or os.getenv("RUN_NAME", f"{layer_type.lower()}-{operation}-in{input_size}-bs{batch_size}")
    notes = args.notes or os.getenv("NOTES", None)

    # Build wrapper command
    repo_root = Path(__file__).resolve().parents[1]
    wrapper = repo_root / "scripts" / "run_wandb_benchmark.py"

    bench_args = [
        "--no-cuda" if no_cuda else "",
        "--layer-type", layer_type,
        "--operation", operation,
        "--input-size", str(input_size),
        "--hidden-size", str(hidden_size),
        "--batch-size", str(batch_size),
        "--max-iterations", str(max_iterations),
        "--log-interval", str(log_interval),
        "--learning-rate", str(learning_rate),
        "--clip-grad-norm", str(clip_grad_norm),
    ]
    bench_args = [a for a in bench_args if a != ""]

    cmd = [
        sys.executable,
        str(wrapper),
        "--name", run_name,
    ]
    if notes:
        cmd += ["--notes", notes]
    cmd += ["--", *bench_args]

    print("Launching RunPod job:")
    print(" ", shlex.join(cmd))
    print(" cwd=", repo_root)

    # Stream output
    proc = subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            print(line, end="")
    finally:
        proc.wait()
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())


