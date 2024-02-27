#!/bin/bash

pip install adapters

CUDA_VISIBLE_DEVICES=0

LANGUAGES=("bg" "ms" "mt")

for LANG in "${LANGUAGES[@]}"; do
    python adapters_hub/examples/pytorch/language-modeling/run_mlm.py \
        --model_name_or_path bert-base-multilingual-cased \
        --train_file "/home/dgurgurov/adapters_mlm/data/wiki/train_wiki_${LANG}.csv" \
        --validation_file "/home/dgurgurov/adapters_mlm/data/wiki/val_wiki_${LANG}.csv" \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 16 \
        --do_train \
        --do_eval \
        --logging_dir "./models/adapters_mlm_wiki/${LANG}/logs" \
        --output_dir "./models/adapters_mlm_wiki/${LANG}" \
        --train_adapter \
        --adapter_config seq_bn \
        --overwrite_output_dir \
        --load_best_model_at_end=True \
        --save_total_limit=1 \
        --evaluation_strategy='steps' \
        --save_strategy='steps' \
        --max_steps=100000 \
        --line_by_line
done

