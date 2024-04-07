#!/bin/bash

pip install adapters

TASK_NAME="adapter_sa"
languages=("Maltese" "Indonesian" "Bulgarian")

for lang in "${languages[@]}"
do

    for i in {1..3}
    do
        python scripts/sa_adapter.py \
            --output_dir "/netscratch/dgurgurov/inject_commonsense/$TASK_NAME/models/$lang/$i" \
            --model_name "bert-base-multilingual-cased" \
            --learning_rate 1e-4 \
            --num_train_epochs 50 \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 32  \
            --evaluation_strategy "epoch" \
            --save_strategy "epoch" \
            --language "$lang"
    done
done
