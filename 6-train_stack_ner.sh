#!/bin/bash

pip install seqeval adapters

TASK_NAME="ner_stack_adapter"

for SOURCE in "cn" "wiki"; do
    for LANGUAGE in "mt" "ms" "bg"; do
        for i in {1..3}; do
            OUTPUT_DIR="/netscratch/dgurgurov/inject_commonsense/$TASK_NAME/$SOURCE/models/$LANGUAGE/$i"
            ADAPTER_DIR="./lang_adapters/$SOURCE/$LANGUAGE"
            ADAPTER_CONFIG="./lang_adapters/$SOURCE/$LANGUAGE/adapter_config.json"

            echo "Training iteration $i for language: $LANGUAGE, source: $SOURCE"
            python scripts/mbert_ner_adapter.py \
                --output_dir $OUTPUT_DIR \
                --model_name "bert-base-multilingual-cased" \
                --adapter_dir $ADAPTER_DIR \
                --adapter_config $ADAPTER_CONFIG \
                --learning_rate 2e-4 \
                --num_train_epochs 100 \
                --per_device_train_batch_size 64 \
                --per_device_eval_batch_size 64 \
                --evaluation_strategy "epoch" \
                --save_strategy "epoch" \
                --weight_decay 0.01 \
                --language $LANGUAGE
        done
    done
done
