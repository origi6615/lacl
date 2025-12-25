# Locally-AdaptiveCompensation-for-AnalyticClass-Incremental-Learning

python main.py LACL --dataset CIFAR-100 --base-ratio 0.5 --phases 5 \
--data-root ~/dataset --IL-batch-size 4096 --num-workers 16 --backbone resnet32 \
--gamma 0.1 --gamma-comp 0.1 --buffer-size 8192 \
--cache-features --backbone-path ./backbone/resnet32_CIFAR-100_0.5_None \
--k 20 --compensation-ratio 1.0 --metric cosine
