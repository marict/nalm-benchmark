#!/bin/bash
# Optional suffix to append to pod names, usage: ./run_experiment.sh mytag
SUFFIX="$1"
if [ -n "$SUFFIX" ]; then
  SUFFIX="-$SUFFIX"
fi

SEED="$2"
# If seed is empty use default 0
if [ -z "$SEED" ]; then
  SEED="--seed 0"
fi

python /Users/paul_curry/ai2/runpod_service/runpod_service.py experiments/single_layer_benchmark.py --layer-type DAG --operation div --input-size 3 --batch-size 256 --max-iterations 300000 --log-interval 10000 --clip-grad-norm 1.0 --lr-cosine --lr-min 1e-4 --pod-name nalm-div$SUFFIX $SEED

python /Users/paul_curry/ai2/runpod_service/runpod_service.py experiments/single_layer_benchmark.py --layer-type DAG --operation mul --input-size 3 --batch-size 256 --max-iterations 300000 --log-interval 10000 --clip-grad-norm 1.0 --lr-cosine --lr-min 1e-4 --pod-name nalm-mul$SUFFIX $SEED

python /Users/paul_curry/ai2/runpod_service/runpod_service.py experiments/single_layer_benchmark.py --layer-type DAG --operation add --input-size 3 --batch-size 256 --max-iterations 300000 --log-interval 10000 --clip-grad-norm 1.0 --lr-cosine --lr-min 1e-4 --pod-name nalm-add$SUFFIX $SEED

python /Users/paul_curry/ai2/runpod_service/runpod_service.py experiments/single_layer_benchmark.py --layer-type DAG --operation sub --input-size 3 --batch-size 256 --max-iterations 300000 --log-interval 10000 --clip-grad-norm 1.0 --lr-cosine --lr-min 1e-4 --pod-name nalm-sub$SUFFIX $SEED

