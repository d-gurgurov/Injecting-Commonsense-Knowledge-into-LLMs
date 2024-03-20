#!/bin/bash

pip install adapters

languages=("Maltese" "Indonesian" "Bulgarian")

for lang in "${languages[@]}"
do

    for i in {1..3}
    do
        python scripts/sa_finetune_adapter.py \
            --output_dir "./models/$lang/$i" \
            --model_name bert-base-multilingual-cased \
            --learning_rate 1e-4 \
            --num_train_epochs 50 \
            --per_device_train_batch_size 64 \
            --per_device_eval_batch_size 64  \
            --evaluation_strategy epoch \
            --save_strategy no \
            --language "$lang"
    done
done
