"""
based on https://huggingface.co/blog/Valerii-Knowledgator/multi-label-classification

Useful for example of multilabel classification.
"""
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments,Trainer
import evaluate
import numpy as np


dataset = load_dataset("knowledgator/events_classification_biotech", trust_remote_code=True)

classes = [class_ for class_ in dataset["train"].features["label 1"].names if class_]
class2id = {class_: id for id, class_ in enumerate(classes)}
id2class = {id: class_ for class_, id in class2id.items()}




model_path = "distilbert-base-uncased" if torch.backends.mps.is_available() else  "microsoft/deberta-v3-small"

tokenizer = AutoTokenizer.from_pretrained(model_path)


def preprocess_function(example):
    text = f"{example['title']}.\n{example['content']}"
    all_labels = example["all_labels"]
    labels = [0.0 for i in range(len(classes))]
    for label in all_labels:
        label_id = class2id[label]
        labels[label_id] = 1.0

    example = tokenizer(text, truncation=True)
    example["labels"] = labels
    return example


tokenized_dataset = dataset.map(preprocess_function)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = sigmoid(predictions)
    predictions = (predictions > 0.5).astype(int).reshape(-1)
    return clf_metrics.compute(
        predictions=predictions, references=labels.astype(int).reshape(-1)
    )


model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    num_labels=len(classes),
    id2label=id2class,
    label2id=class2id,
    problem_type="multi_label_classification",
)


training_args = TrainingArguments(
    output_dir="my_awesome_model",
    learning_rate=2e-5,
    per_device_train_batch_size=3,
    per_device_eval_batch_size=3,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


if __name__ == "__main__":
    trainer.train()

