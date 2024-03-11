from dataclasses import dataclass

from transformers import AutoTokenizer,AutoModelForMultipleChoice
import torch
import datasets
from collators import DataCollatorForMultipleChoice



case_hold = datasets.load_dataset("lex_glue","case_hold", trust_remote_code=True)

model_path = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_path,use_fast=True)

def casehold_preprocess_function(examples):
    """
    Produces a collection of five token sequences, one per alternative, with the tokenizer
    merging the contexts and the holdings as desired
    """
    contexts = [[context] * 5 for context in examples["context"]]

    contexts = sum(contexts, [])

    holdings = sum(examples['endings'], [])

    tokenized_examples = tokenizer(contexts, holdings, truncation=True)
    return {k: [v[i: i + 5] for i in range(0, len(v), 5)] for k, v in tokenized_examples.items()}


features = casehold_preprocess_function(case_hold['train'][:10])


collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)
batch = collator(features=features)
print(batch)

