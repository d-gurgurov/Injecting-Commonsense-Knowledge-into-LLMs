#!/bin/bash

pip install adapters

languages=("Maltese" "Indonesian" "Bulgarian")

for lang in "${languages[@]}"
do

    for i in {1..3}
    do
        output_dir="$i"
        logging_dir="$i/logs"

        python scripts/sa_fusion.py --adapter_cn_dir "./lang_adapters/cn/$lang" \
                                          --adapter_wiki_dir "./lang_adapters/wiki/$lang" \
                                          --lang_adapter_cn "./lang_adapters/cn/$lang/adapter_config.json" \
                                          --lang_adapter_wiki "./lang_adapters/wiki/$lang/adapter_config.json" \
                                          --output_dir "./models/$lang/$output_dir" \
                                          --learning_rate 1e-4 \
                                          --language "$lang"
    done
done
