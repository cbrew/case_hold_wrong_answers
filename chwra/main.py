"""
Trainer file for case_hold.
"""
import os
from datasets import load_dataset
import wandb
import evaluate

from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import torch
from transformers import AutoTokenizer
import numpy as np
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer



accuracy = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")



# put your wandb key in an env var WANDB_API_KEY, but don't check it in, for security.
os.environ["WANDB_PROJECT"] = "case_hold_wrong_answers"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"
wandb.login()

run = wandb.init(
    # Set the project where this run will be logged
    project="case_hold_wrong_answers",
)
case_hold = load_dataset("lex_glue", "case_hold")


model = "sentence-transformers/all-MiniLM-L6-v2"
save = "all_minilm_l6_v2"


tokenizer = AutoTokenizer.from_pretrained(model)


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


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()}
             for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {**accuracy.compute(predictions=predictions, references=labels),
            **f1_metric.compute(predictions=predictions, references=labels,average="macro")}

model = AutoModelForMultipleChoice.from_pretrained(model)

training_args = TrainingArguments(
    output_dir=save,
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
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