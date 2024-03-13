#!/bin/bash

# List of languages
languages=("Maltese" "Indonesian" "Bulgarian")

# Loop over languages
for lang in "${languages[@]}"; do
    echo "Training for language: $lang"

    # Fine-tuning command with language-specific parameters
    python scripts/mbert_sa.py \
        --output_dir "./models/$lang/" \
        --model_name bert-base-multilingual-cased \
        --learning_rate 1e-4 \
        --num_train_epochs 50 \
        --per_device_train_batch_size 64 \
        --per_device_eval_batch_size 64  \
        --evaluation_strategy steps \
        --save_strategy steps \
        --logging_dir "./models/$lang/logs/" \
	--language "$lang"

    echo "Training for $lang completed."
done

