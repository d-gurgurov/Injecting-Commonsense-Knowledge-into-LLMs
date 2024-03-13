#!/bin/bash

pip install adapters

# fusing two mlm adapters trained with different data types and then fine-tuning for classification task

for i in {1..3}
do
    output_dir="$i"
    logging_dir="$i/logs"
    language='mt'

    python scripts/adhub_fusion_sa.py --adapter_cn_dir ./lang_adapters/cn/"$language" \
                                      --adapter_wiki_dir ./lang_adapters/wiki/"$language" \
                                      --lang_adapter_cn ./lang_adapters/cn/"$language"/adapter_config.json \
                                      --lang_adapter_wiki ./lang_adapters/wiki/"$language"/adapter_config.json \
                                      --output_dir ./models/"$language"/"$output_dir" \
                                      --logging_dir ./models/"$language"/"$logging_dir" \
                                      --learning_rate 1e-4

done
