python /Users/paul_curry/ai2/runpod_service/runpod_service.py experiments/single_layer_benchmark.py --layer-type DAG --operation div --input-size 3 --batch-size 256 --max-iterations 25000 --log-interval 100 --clip-grad-norm 1.0 --pod-name nalm-div

python /Users/paul_curry/ai2/runpod_service/runpod_service.py experiments/single_layer_benchmark.py --layer-type DAG --operation mul --input-size 3 --batch-size 256 --max-iterations 25000 --log-interval 100 --clip-grad-norm 1.0 --pod-name nalm-mul

python /Users/paul_curry/ai2/runpod_service/runpod_service.py experiments/single_layer_benchmark.py --layer-type DAG --operation add --input-size 3 --batch-size 256 --max-iterations 25000 --log-interval 100 --clip-grad-norm 1.0 --pod-name nalm-add

python /Users/paul_curry/ai2/runpod_service/runpod_service.py experiments/single_layer_benchmark.py --layer-type DAG --operation sub --input-size 3 --batch-size 256 --max-iterations 25000 --log-interval 100 --clip-grad-norm 1.0 --pod-name nalm-sub

