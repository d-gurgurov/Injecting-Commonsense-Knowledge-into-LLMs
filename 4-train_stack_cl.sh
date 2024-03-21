#!/bin/bash

pip install adapters

TASK_NAME="stack_sa"

languages=("Maltese" "Indonesian" "Bulgarian")

adapter_sources=("wiki" "cn")

for source in "${adapter_sources[@]}"; do
    echo "Using adapter source: $source"
    
    for lang in "${languages[@]}"; do
        echo "Training for language: $lang"
        
        for i in {1..3}; do
            output_dir="/netscratch/dgurgurov/inject_commonsense/$TASK_NAME/${lang}_${source}/$i"
            adapter_dir="./lang_adapters/${source}/$lang"
            adapter_config="./lang_adapters/${source}/$lang/adapter_config.json"
            
            echo "Iteration: $i, Output Directory: $output_dir"
            
            python scripts/sa_finetune_llm.py \
                --output_dir "$output_dir" \
                --adapter_dir "$adapter_dir" \
                --adapter_config "$adapter_config" \
                --model_name bert-base-multilingual-cased \
                --learning_rate 1e-4 \
                --num_train_epochs 50 \
                --per_device_train_batch_size 64 \
                --per_device_eval_batch_size 64 \
                --evaluation_strategy epoch \
                --save_strategy epoch \
                --language "$lang"
        done
        
        echo "Training for $lang completed."
    done
done
