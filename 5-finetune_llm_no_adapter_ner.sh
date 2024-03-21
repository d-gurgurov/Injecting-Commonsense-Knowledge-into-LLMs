#!/bin/bash

TASK_NAME="llm_no_adapter_ner"
LANGUAGES=("mt" "ms" "bg")

pip install seqeval

for LANGUAGE in "${LANGUAGES[@]}"; do
    for i in {1..3}; do
        OUTPUT_DIR="/netscratch/dgurgurov/inject_commonsense/$TASK_NAME/models/$LANGUAGE/$i"

        echo "Training iteration $i for language: $LANGUAGE"
        python scripts/mbert_ner.py \
            --output_dir $OUTPUT_DIR \
            --model_name "bert-base-multilingual-cased" \
            --learning_rate 2e-5 \
            --num_train_epochs 100 \
            --per_device_train_batch_size 64 \
            --per_device_eval_batch_size 64 \
            --evaluation_strategy "epoch" \
            --save_strategy "epoch" \
            --weight_decay 0.01 \
            --language $LANGUAGE
    done
done
