#!/bin/bash

CUDA_VISIBLE_DEVICES=0

LANGUAGES=("bg" "ms" "mt")
DATA_SOURCES=("wiki" "cn")

for LANG in "${LANGUAGES[@]}"; do
    for DATA_SRC in "${DATA_SOURCES[@]}"; do
        python adapters/examples/pytorch/language-modeling/run_mlm.py \
            --model_name_or_path bert-base-multilingual-cased \
            --train_file "./data/${DATA_SRC}/train_${DATA_SRC}_${LANG}.csv" \
            --validation_file "./data/${DATA_SRC}/val_${DATA_SRC}_${LANG}.csv" \
            --per_device_train_batch_size 16 \
            --per_device_eval_batch_size 16 \
            --do_train \
            --do_eval \
            --logging_dir "./models/adapters_mlm_${DATA_SRC}/${LANG}/logs" \
            --output_dir "./models/adapters_mlm_${DATA_SRC}/${LANG}" \
            --train_adapter \
            --adapter_config seq_bn \
            --overwrite_output_dir \
            --load_best_model_at_end=True \
            --save_total_limit=1 \
            --evaluation_strategy='steps' \
            --save_strategy='steps' \
            --max_steps=50000 \
            --line_by_line
    done
done


