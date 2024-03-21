#!/bin/bash

TASK_NAME="sa_no_adapter"
languages=("Maltese" "Indonesian" "Bulgarian")

for lang in "${languages[@]}"; do
    echo "Training for language: $lang"

    for i in {1..3}; do
        output_dir="/netscratch/dgurgurov/inject_commonsense/$TASK_NAME/models/$lang/$i/"
        echo "Iteration: $i, Output Directory: $output_dir"

        python scripts/sa_finetune_mbert.py \
            --output_dir "$output_dir" \
            --model_name bert-base-multilingual-cased \
            --learning_rate 2e-5 \
            --num_train_epochs 50 \
            --per_device_train_batch_size 64 \
            --per_device_eval_batch_size 64  \
            --evaluation_strategy epoch \
            --save_strategy epoch \
            --language "$lang"

        echo "Training for $lang (iteration $i) completed."
    done

    echo "Training for $lang completed."
done
