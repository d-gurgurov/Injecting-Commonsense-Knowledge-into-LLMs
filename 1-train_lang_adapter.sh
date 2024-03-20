#!/bin/bash

pip install adapters
pip install evaluate

languages=("mt" "bg" "ms")
adapter_sources=("wiki" "cn")

for source in "${adapter_sources[@]}"; do
    echo "Using adapter source: $source"
    
    for lang in "${languages[@]}"; do
        echo "Training for language: $lang"
        
        python adapters_hub/examples/pytorch/language-modeling/run_mlm.py \
            --model_name_or_path bert-base-multilingual-cased \
            --train_file "/home/dgurgurov/adapters_mlm/data/$source/train_${source}_${lang}.csv" \
            --validation_file "/home/dgurgurov/adapters_mlm/data/$source/val_${source}_${lang}.csv" \
            --per_device_train_batch_size 16 \
            --per_device_eval_batch_size 16 \
            --do_train \
            --do_eval \
            --logging_dir "./models/adapters_mlm_${source}/${lang}/logs" \
            --output_dir "./models/adapters_mlm_${source}/${lang}" \
            --train_adapter \
            --adapter_config seq_bn \
            --overwrite_output_dir \
            --load_best_model_at_end=True \
            --save_total_limit=1 \
            --evaluation_strategy='steps' \
            --save_strategy='steps' \
            --max_steps=50000
        
        echo "Training for $lang completed."
    done
done
