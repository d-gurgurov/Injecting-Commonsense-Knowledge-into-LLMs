# import dependencies
import argparse
import evaluate
import numpy as np
import os
import re
import sys
import json
import torch
import transformers

from datasets import load_dataset
from transformers import (
    DataCollatorForLanguageModeling,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    BertForSequenceClassification,
    AutoConfig,
    BertConfig,
    Trainer,
    TrainingArguments,
)


def encode_batch(examples, tokenizer):
    """Encodes a batch of input data using the model tokenizer."""
    all_encoded = {"input_ids": [], "attention_mask": [], "labels": []}
    
    for text, label in zip(examples["text"], examples["label"]):
        encoded = tokenizer(
            text,
            max_length=512,
            truncation=True,
            padding="max_length",
        )
        all_encoded["input_ids"].append(encoded["input_ids"])
        all_encoded["attention_mask"].append(encoded["attention_mask"])
        all_encoded["labels"].append(label)
    
    return all_encoded


def preprocess_dataset(dataset, tokenizer):
    dataset = dataset.map(lambda sample: encode_batch(sample, tokenizer), batched=True)
    dataset.set_format(columns=["input_ids", "attention_mask", "labels"])
    return dataset

def parse_arguments():
    parser = argparse.ArgumentParser(description="Fine-tune a model for a sentiment analysis task.")
    parser.add_argument("--output_dir", type=str, default="./training_output", help="Output directory for training results")
    parser.add_argument("--logging_dir", type=str, default="./models/logs", help="Output directory for log results")
    parser.add_argument("--model_name", type=str, default="bert-base-multilingual-cased", help="Name of the pre-trained model")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for training")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=32, help="Batch size per device during training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32, help="Batch size per device during evaluation")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging frequency during training")
    parser.add_argument("--evaluation_strategy", type=str, default="epoch", help="Evaluation strategy during training")
    parser.add_argument("--save_strategy", type=str, default="epoch", help="Saving strategy during training")
    parser.add_argument("--save_steps", type=int, default=10, help="Saving frequency during training")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio for learning rate scheduler")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimization")
    parser.add_argument("--language", type=str, help='Language for fine-tuning')
    return parser.parse_args()


def calculate_f1_on_test_set(trainer, test_dataset):
    print("Calculating F1 score on the test set...")
    test_predictions = trainer.predict(test_dataset)

    # Compute F1 score on the test set
    f1_metric = evaluate.load("f1")
    test_metrics = {
        "f1": f1_metric.compute(
            predictions=np.argmax(test_predictions.predictions, axis=-1),
            references=test_predictions.label_ids,
            average="macro",
        )["f1"],
    }

    print("Test F1 score:", test_metrics["f1"])

    return test_metrics


def main():
    args = parse_arguments()

    dataset = load_dataset(f"sepidmnorozy/{args.language}_sentiment")

    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    test_dataset = dataset["test"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_dataset = preprocess_dataset(train_dataset, tokenizer)
    val_dataset = preprocess_dataset(val_dataset, tokenizer)
    test_dataset = preprocess_dataset(test_dataset, tokenizer)

    config = AutoConfig.from_pretrained(args.model_name)
    model = BertForSequenceClassification(config=config)
    model.config.hidden_dropout_prob = 0.5
    model.config.attention_probs_dropout_prob = 0.5

    print(model.config)
    training_args = TrainingArguments(
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        logging_steps=args.logging_steps,
	logging_dir=args.logging_dir,
        save_strategy=args.save_strategy,
        evaluation_strategy=args.evaluation_strategy,
        save_steps=args.save_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        save_total_limit=1,
        load_best_model_at_end=True,
        remove_unused_columns=False,
    )
    
    f1_metric = evaluate.load("f1")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=lambda pred: {
                "f1": f1_metric.compute(
                    predictions=np.argmax(pred.predictions, axis=-1),
                    references=pred.label_ids,
                    average="macro",
                )["f1"],
            },
    )

    trainer.train()

    calculate_f1_on_test_set(trainer, test_dataset)

    output_file_path = os.path.join(args.output_dir, "test_metrics.json")
    with open(output_file_path, "w") as json_file:
        json.dump(calculate_f1_on_test_set(trainer, test_dataset), json_file, indent=2)


if __name__ == "__main__":
    main()
