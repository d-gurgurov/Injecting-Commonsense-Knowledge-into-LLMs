#!/bin/bash

pip install adapters

for i in {1..3}
do
    python scripts/adhub_mbert_sa.py \
        --output_dir ./models/ms/$i \
        --model_name bert-base-multilingual-cased \
        --learning_rate 1e-4 \
        --num_train_epochs 50 \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 32  \
        --evaluation_strategy steps \
        --save_strategy steps \
        --logging_dir ./models/ms/$i/logs
done
