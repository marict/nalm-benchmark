# Benchmark settings and example commands

This document collects practical command-lines to reproduce common single-module arithmetic benchmarks used across NALM papers, using this repository's `experiments/single_layer_benchmark.py` entry point. These are reference command templates; papers typically run long sweeps over seeds and ranges (see notes at the end).

## Environment

```bash
conda env create -f nalu-env.yml
conda activate nalu-env
python3 setup.py develop
# or, system Python:
# python3 -m pip install torch torchvision scipy pandas tensorboard tensorboardX tensorflow-macos
```

## General single-run template

```bash
python3 experiments/single_layer_benchmark.py \
  --layer-type <UNIT> \
  --operation <add|sub|mul|div> \
  --input-size <N> \
  --hidden-size 2 \
  --num-subsets 2 \
  --interpolation-range "[1,2]" \
  --extrapolation-range "[2,6]" \
  --batch-size 128 \
  --max-iterations 500000 \
  --seed 0 \
  --log-interval 10000
```

Notes
- CPU only: add `--no-cuda`.
- Increase `--max-iterations` to match paper-length runs (millions of steps for some figures).
- For reproducible figure plots, run many seeds (see sweep notes below).

## Unit name mapping (paper → repo)

- NAC+ → `--layer-type NAC`
- NAC* → `--layer-type NAC --nac-mul normal`
- NALU → `--layer-type NALU`
- NAU → `--layer-type ReRegualizedLinearNAC`
- NMU → `--layer-type ReRegualizedLinearNAC --nac-mul mnac`
- iNALU → `--layer-type iNALU` (optionally use `--reinit` flags)

## Primer figures (single-module arithmetic)

These templates mirror common settings (two subsets, ranges [1,2] and [2,6]). Adjust `--max-iterations` and run multiple seeds.

### Addition
```bash
# NAU (addition)
python3 experiments/single_layer_benchmark.py \
  --layer-type ReRegualizedLinearNAC \
  --operation add --input-size 2 --hidden-size 2 \
  --batch-size 128 --max-iterations 500000 --seed 0

# NMU (addition)
python3 experiments/single_layer_benchmark.py \
  --layer-type ReRegualizedLinearNAC --nac-mul mnac \
  --operation add --input-size 2 --hidden-size 2 \
  --batch-size 128 --max-iterations 500000 --seed 0

# NALU (addition)
python3 experiments/single_layer_benchmark.py \
  --layer-type NALU \
  --operation add --input-size 2 --hidden-size 2 \
  --batch-size 128 --max-iterations 500000 --seed 0

# DAG (this repo's DAG layer)
python3 experiments/single_layer_benchmark.py \
  --layer-type DAG \
  --operation add --input-size 4 --hidden-size 2 \
  --batch-size 128 --max-iterations 500000 --seed 0
```

### Subtraction
```bash
python3 experiments/single_layer_benchmark.py --layer-type NALU --operation sub --input-size 2 --hidden-size 2 --batch-size 128 --max-iterations 500000 --seed 0
python3 experiments/single_layer_benchmark.py --layer-type ReRegualizedLinearNAC --operation sub --input-size 2 --hidden-size 2 --batch-size 128 --max-iterations 500000 --seed 0
python3 experiments/single_layer_benchmark.py --layer-type DAG --operation sub --input-size 4 --hidden-size 2 --batch-size 128 --max-iterations 500000 --seed 0
```

### Multiplication
```bash
python3 experiments/single_layer_benchmark.py --layer-type NALU --operation mul --input-size 2 --hidden-size 2 --batch-size 128 --max-iterations 500000 --seed 0
python3 experiments/single_layer_benchmark.py --layer-type ReRegualizedLinearNAC --nac-mul mnac --operation mul --input-size 2 --hidden-size 2 --batch-size 128 --max-iterations 500000 --seed 0
python3 experiments/single_layer_benchmark.py --layer-type DAG --operation mul --input-size 4 --hidden-size 2 --batch-size 128 --max-iterations 500000 --seed 0
```

### Division
```bash
python3 experiments/single_layer_benchmark.py --layer-type NALU --operation div --input-size 2 --hidden-size 2 --batch-size 128 --max-iterations 500000 --seed 0
python3 experiments/single_layer_benchmark.py --layer-type ReRegualizedLinearNAC --operation div --input-size 2 --hidden-size 2 --batch-size 128 --max-iterations 500000 --seed 0
python3 experiments/single_layer_benchmark.py --layer-type DAG --operation div --input-size 4 --hidden-size 2 --batch-size 128 --max-iterations 500000 --seed 0
```

## iNALU (input size 10)

```bash
python3 experiments/single_layer_benchmark.py \
  --layer-type iNALU --operation add \
  --input-size 10 --hidden-size 2 \
  --batch-size 128 --max-iterations 1000000 \
  --seed 0 --reinit --reinit-epoch-interval 10 --reinit-max-stored-losses 5000 --reinit-loss-thr 1.0
```

## NAU (input size 100)

```bash
python3 experiments/single_layer_benchmark.py \
  --layer-type ReRegualizedLinearNAC \
  --operation sub --input-size 100 --hidden-size 2 \
  --batch-size 128 --max-iterations 1000000 --seed 0
```

## Sweep notes (to match paper plots)

- Seeds: run 25 seeds (0–24). Example:
  ```bash
  for s in $(seq 0 24); do \
    python3 experiments/single_layer_benchmark.py --layer-type NALU --operation add --input-size 2 --hidden-size 2 --batch-size 128 --max-iterations 500000 --seed $s &
  done; wait
  ```
- Export to CSV: `python3 export/simple_function_static.py --tensorboard-dir tensorboard/<exp_name>/ --csv-out results.csv`
- Plot: use the R scripts in `export/single_layer_task/benchmark/`.

## Tips for the DAG layer

- CPU run example (stable settings):
  ```bash
  python3 experiments/single_layer_benchmark.py --no-cuda --layer-type DAG --operation add --input-size 4 --hidden-size 2 --batch-size 64 --max-iterations 300000 --learning-rate 1e-4 --clip-grad-norm 1.0
  ```
- Depth is internal to the layer (defaults to `in_features-1`).
- At eval the executor uses hard O/G (rounded/clamped) for discrete plans.

## Run on RunPod with Weights & Biases

Use the provided wrapper to stream metrics (train/inter/extra) to wandb. The project and entity are fixed to `nalm-benchmark` and `paul-michael-curry-productions`.

Example (multiplication):

```bash
python3 scripts/run_wandb_benchmark.py \
  --name dag-mul-in4-bs1000 \
  -- --no-cuda --layer-type DAG --operation mul \
  --input-size 4 --hidden-size 2 \
  --batch-size 1000 --max-iterations 300000 \
  --log-interval 1000 --learning-rate 1e-4 --clip-grad-norm 1.0
```

Notes for RunPod:
- Ensure WANDB_API_KEY is set in the environment.
- The wrapper parses stdout lines printed by the benchmark (e.g., `train X: a, inter: b, extra: c`) and logs to wandb.

### RunPod service (arg-based)

Alternatively, launch via the dedicated service script (reads args or env):

```bash
python3 scripts/runpod_launcher.py \
  --wandb-api-key "$WANDB_API_KEY" \
  --run-name dag-mul-in4-bs1000 \
  --no-cuda \
  --layer-type DAG \
  --operation mul \
  --input-size 4 --hidden-size 2 \
  --batch-size 1000 --max-iterations 300000 \
  --log-interval 1000 --learning-rate 1e-4 --clip-grad-norm 1.0
```


