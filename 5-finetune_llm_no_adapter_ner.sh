#!/bin/bash

LEARNING_RATE=2e-5
NUM_TRAIN_EPOCHS=50
PER_DEVICE_TRAIN_BATCH_SIZE=64
PER_DEVICE_EVAL_BATCH_SIZE=64
EVALUATION_STRATEGY="epoch"
SAVE_STRATEGY="epoch"
WEIGHT_DECAY=0.01

pip install seqeval

for i in {1..3}
do
    OUTPUT_DIR="./models/ms/$i"
    LOGGING_DIR="./models/ms/$i/logs"

    echo "Training iteration $i"
    python scripts/ner_no_adapter.py \
        --output_dir $OUTPUT_DIR \
        --logging_dir $LOGGING_DIR \
        --model_name "bert-base-multilingual-cased" \
        --learning_rate $LEARNING_RATE \
        --num_train_epochs $NUM_TRAIN_EPOCHS \
        --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
        --per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE \
        --evaluation_strategy $EVALUATION_STRATEGY \
        --save_strategy $SAVE_STRATEGY \
        --weight_decay $WEIGHT_DECAY \
        --language "ms"
done
