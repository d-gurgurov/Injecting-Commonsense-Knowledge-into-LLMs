#!/bin/bash

pip install adapters

for i in {1..3}
do
    python scripts/adhub_mbert_mlm_sa.py \
        --output_dir ./models_cl/ms_wiki/$i \
        --adapter_dir ./models/adapters_mlm_wiki/ms \
        --logging_dir ./models_cl/ms_wiki/$i/logs \
        --adapter_config ./models/adapters_mlm_wiki/ms/adapter_config.json \
        --model_name bert-base-multilingual-cased \
        --learning_rate 1e-4 \
        --num_train_epochs 50 \
        --per_device_train_batch_size 64 \
        --per_device_eval_batch_size 64  \
        --evaluation_strategy steps \
        --save_strategy steps
done
