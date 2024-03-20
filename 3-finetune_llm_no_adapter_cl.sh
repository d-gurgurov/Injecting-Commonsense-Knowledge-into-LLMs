#!/bin/bash

languages=("Maltese" "Indonesian" "Bulgarian")

for lang in "${languages[@]}"; do
    echo "Training for language: $lang"

    for i in {1..3}; do
        output_dir="./models/$lang/$i/"
        echo "Iteration: $i, Output Directory: $output_dir"

        python scripts/sa_no_adapter.py \
            --output_dir "$output_dir" \
            --model_name bert-base-multilingual-cased \
            --learning_rate 1e-4 \
            --num_train_epochs 50 \
            --per_device_train_batch_size 64 \
            --per_device_eval_batch_size 64  \
            --evaluation_strategy epoch \
            --save_strategy no \
            --language "$lang"

        echo "Training for $lang (iteration $i) completed."
    done

    echo "Training for $lang completed."
done
