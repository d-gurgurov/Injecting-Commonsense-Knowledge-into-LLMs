import re
import csv
import random
from datasets import Dataset
from datasets import load_dataset

def process_relationships_dataset(dataset_dict):
    """
    Processes a DatasetDict object with CSV files containing relationships and constructs datasets
    suitable for masked language modeling tasks, using regular expressions to
    extract the relationship string.

    Args:
        dataset_dict (DatasetDict): A DatasetDict object containing paths to CSV files with relationships.

    Returns:
        DatasetDict: A DatasetDict object with processed datasets, where each split contains 'sentence' and 'word' columns.
    """

    relationships = [
        'is the opposite of',
        'is derived from',
        'is etymologically derived from',
        'is etymologically related to',
        'is a form of',
        'has context of',
        'is a type of',
        'is related to',
        'is similar to',
        'is a synonym of',
        'is a symbol of',
        'is distinct from',
    ]

    pattern = rf"(.*?)({'|'.join(relationships)})(.+)"  # Regular expression pattern

    for split, dataset in dataset_dict.items():
        data = {'sentence': [], 'word': []}
        for text in dataset['text']:
            match = re.search(pattern, text)
            if match:
                # Extract sentence parts before and after the relationship
                sentence_before = match.group(1).strip()
                sentence_after = match.group(3).strip(".")

                # Randomly choose whether to use sentence before or after
                if random.random() < 0.5:
                    word_to_mask = sentence_before
                else:
                    word_to_mask = sentence_after

                data['sentence'].append(text)
                data['word'].append(word_to_mask)

        dataset_dict[split] = Dataset.from_dict(data)

    return dataset_dict

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

MASK_TOKEN = tokenizer.mask_token
MASK_TOKEN_ID = tokenizer.mask_token_id

def tokenize_function(examples):
    tokenized_sentences = tokenizer(examples['sentence'], padding="max_length", truncation=True, max_length=128)

    label_arr_list = []

    for i, (sentence, word) in enumerate(zip(examples['sentence'], examples['word'])):
        words_to_replace = word.split()
        sentence_tokens = tokenized_sentences.input_ids[i]
        attention_mask = tokenized_sentences.attention_mask[i]

        label_arr = [-100] * len(sentence_tokens)

        for w in words_to_replace:
            tokens_ids = tokenizer.encode(w, add_special_tokens=False)
            word_tokens = tokens_ids
            start_index = 0
            while start_index < len(sentence_tokens):
                try:
                    idx = sentence_tokens.index(word_tokens[0], start_index)
                    if sentence_tokens[idx:idx+len(word_tokens)] == word_tokens:
                        # Replace the tokens corresponding to the word with MASK_TOKEN_ID
                        sentence_tokens[idx:idx+len(word_tokens)] = [MASK_TOKEN_ID] * len(word_tokens)
                        # Mark these tokens as not ignored for loss computation
                        label_arr[idx:idx+len(word_tokens)] = word_tokens
                        # Move start_index to the end of the replaced word
                        start_index = idx + len(word_tokens)
                    else:
                        start_index += 1
                except ValueError:
                    break

        # Ensure that the attention mask is updated accordingly
        attention_mask = [1 if token != tokenizer.pad_token_id else 0 for token in sentence_tokens]

        tokenized_sentences.input_ids[i] = sentence_tokens
        tokenized_sentences.attention_mask[i] = attention_mask

        label_arr_list.append(label_arr)

    # Prepare the output dictionary
    output_dict = {
        'input_ids': tokenized_sentences.input_ids,
        'attention_mask': tokenized_sentences.attention_mask,
        'labels': label_arr_list
    }

    return output_dict


data_files = {"train": "train_cn_mt.csv", "test": "val_cn_mt.csv"}
dataset = load_dataset("mlm_task/clean_cn", data_files=data_files)

processed_dataset_dict = process_relationships_dataset(dataset)

processed_dataset_dict.map(tokenize_function, batched=True)