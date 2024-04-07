#!/bin/bash

TASK_NAME="fusion_ner"
LANGUAGES=("bg")
ADAPTER_CN_DIR="./lang_adapters/cn"
ADAPTER_WIKI_DIR="./lang_adapters/wiki"

pip install seqeval adapters

for LANGUAGE in "${LANGUAGES[@]}"; do
    for i in {1..3}; do
        OUTPUT_DIR="/netscratch/dgurgurov/inject_commonsense/fullwordmasking/$TASK_NAME/models/$LANGUAGE/$i"
        ADAPTER_CN_CONFIG="./lang_adapters/cn/$LANGUAGE/adapter_config.json"
        ADAPTER_WIKI_CONFIG="./lang_adapters/wiki/$LANGUAGE/adapter_config.json"

        echo "Training iteration $i for language: $LANGUAGE"
        python scripts/ner_fusion.py \
            --output_dir $OUTPUT_DIR \
            --model_name "bert-base-multilingual-cased" \
            --adapter_cn_dir $ADAPTER_CN_DIR/$LANGUAGE \
            --adapter_wiki_dir $ADAPTER_WIKI_DIR/$LANGUAGE \
            --adapter_cn_config $ADAPTER_CN_CONFIG \
            --adapter_wiki_config $ADAPTER_WIKI_CONFIG \
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
