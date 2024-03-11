"""
Trainer file for case_hold.
"""
import os
from datasets import load_dataset
import wandb
import evaluate

import torch
from transformers import AutoTokenizer
import numpy as np
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer
import torch.nn.functional as F

from chwra.collators import DataCollatorForMultipleChoice

accuracy = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")


ckpt = "distilbert-base-uncased"
save = "distilbert_base_uncased"


# put your wandb key in an env var WANDB_API_KEY, but don't check it in, for security.
os.environ["WANDB_PROJECT"] = "case_hold_wrong_answers"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"
wandb.login()
run = wandb.init(
        # Set the project where this run will be logged
        project="case_hold_wrong_answers",
)
case_hold = load_dataset("lex_glue", "case_hold")




tokenizer = AutoTokenizer.from_pretrained(ckpt)


def preprocess_function(examples):
    """
    Produces a collection of five token sequences, one per alternative, with the tokenizer
    merging the contexts and the holdings as desired
    """
    contexts = [[context] * 5 for context in examples["context"]]

    contexts = sum(contexts, [])

    holdings = sum(examples['endings'], [])

    tokenized_examples = tokenizer(contexts, holdings, truncation=True)
    return {k: [v[i: i + 5] for i in range(0, len(v), 5)] for k, v in tokenized_examples.items()}



tokenized_casehold = case_hold.map(preprocess_function, batched=True)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {**accuracy.compute(predictions=predictions, references=labels),
            **f1_metric.compute(predictions=predictions, references=labels,average="macro")}


model = AutoModelForMultipleChoice.from_pretrained(ckpt)


training_args = TrainingArguments(
    output_dir=save,
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=5,
    load_best_model_at_end=True,
    fp16 = not(torch.backends.mps.is_available()), # checks if we are on a Mac.
    learning_rate=1e-6,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=15,
    weight_decay=0.01,
    push_to_hub=False,

)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_casehold["train"],
    eval_dataset=tokenized_casehold["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)


if __name__ == "__main__":
    trainer.train()
    if wandb.run is not None:
        wandb.finish()