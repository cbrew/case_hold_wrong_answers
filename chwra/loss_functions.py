import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss,CrossEntropyLoss
import torch.nn.functional as F
from datasets import load_dataset
import collators
from torch.utils.data.dataloader import DataLoader

case_hold = load_dataset("lex_glue","case_hold")

from transformers import BertTokenizer,BertModel

val_data = case_hold["validation"]

tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
def preprocess_fn(examples):
    """
    Produces a collection of five token sequences, one per alternative, with the tokenizer
    merging the contexts and the holdings as desired
    """
    contexts = [[context] * 5 for context in examples["context"]]

    contexts = sum(contexts, [])

    holdings = sum(examples["endings"], [])

    tokenized_examples = tokenizer(
        contexts, holdings, truncation=True, max_length=512
    )
    features = {
        k: [v[i: i + 5] for i in range(0, len(v), 5)]
        for k, v in tokenized_examples.items()
    }
    return features

tokenized = val_data.map(preprocess_fn,batched=True,batch_size=12)

model = BertModel.from_pretrained("google-bert/bert-base-uncased")
classifier_head = nn.Linear(768,1)

data_loader = DataLoader(tokenized,batch_size=23,collate_fn=collators.DataCollatorForMultipleChoice(tokenizer))

for x in data_loader:
    input_ids = x['input_ids']
    attention_mask = x['attention_mask']
    input_ids = input_ids.view(-1, input_ids.size(-1))
    attention_mask = attention_mask.view(-1, attention_mask.size(-1))

    outputs = model(input_ids=input_ids,attention_mask=attention_mask)
    logits = classifier_head(outputs.pooler_output)


    # how to calculate the wrong answer loss.
    # turn it into a flat collection of labels with 1s
    # everywhere that there is a wrong answer.
    labels = 1 - F.one_hot(x['labels']).float().reshape(-1,1)

    loss_fcn = BCEWithLogitsLoss()

    with torch.no_grad():
        print("BCE loss",loss_fcn(logits,labels))
    print(logits)
    print(labels)

    break