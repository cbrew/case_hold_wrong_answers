"""
Minimal deno of multiple choice for case hold using lightning.
"""

import functools
import os
from argparse import ArgumentParser


from pytorch_lightning import LightningModule, Trainer
from transformers import (
    DistilBertTokenizer,
    DistilBertForMultipleChoice,
    DistilBertModel
)
from torch.utils.data.dataloader import DataLoader
import datasets
import torch
from torch import nn
from lightning.pytorch.loggers import WandbLogger,Logger
from torchmetrics.functional import accuracy

from chwra.collators import DataCollatorForMultipleChoice
import wandb

class MyMultipleChoice(nn.Module):
    """
    Mirrors DistilBertForMultipleChoice.
    """
    def __init__(self):
        super().__init__()
        self.dim = 768
        self.ckpt = "distilbert-base-uncased"
        self.distilbert = DistilBertModel.from_pretrained(self.ckpt)
        self.pre_classifier = nn.Linear(self.dim,self.dim)
        self.classifier = nn.Linear(self.dim, 1)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self,input_ids,attention_mask):
        # reshape input ids and attention mask
        num_choices = input_ids.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        outputs = self.distilbert(input_ids,attention_mask=attention_mask)
        hidden_state = outputs[0]
        pooled_output = hidden_state[:, 0]
        pooled_output = self.pre_classifier(pooled_output)  # (bs * num_choices, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs * num_choices, dim)
        pooled_output = self.dropout(pooled_output)  # (bs * num_choices, dim)
        logits = self.classifier(pooled_output)  # (bs * num_choices, 1)
        reshaped_logits = logits.view(-1, num_choices)  # (bs, num_choices)
        return reshaped_logits



        # undo reshaping



class DistilBertFineTune(LightningModule):
    """"
    Fine tuning module for distilbert multiple choice
    """
    def __init__(self):
        super().__init__()
        self.distilbert = MyMultipleChoice()
        self.ckpt = self.distilbert.ckpt
        self.loss_fct = nn.CrossEntropyLoss()




    def training_step(self, batch, batch_idx):

        labels = batch["labels"]
        logits = self.distilbert(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )

        loss = self.loss_fct(logits, labels)
        self.log("train_loss", loss)

        preds = logits.argmax(dim=1)
        acc = accuracy(preds, labels, task="multiclass", num_classes=5)
        self.log("train_accuracy", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch["labels"]
        logits = self.distilbert(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        loss = self.loss_fct(logits, labels)
        self.log("eval_loss", loss)

        preds = logits.argmax(dim=1)
        acc = accuracy(preds, labels,task="multiclass",num_classes=5)
        self.log("eval_accuracy", acc)
        return preds


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-6)
        return optimizer




def main(hparams):
    """Main function as suggeested in documentation for Trainer"""
    # recommended incantation to make good use of tensor cores.


    torch.set_float32_matmul_precision('medium')
    case_hold = datasets.load_dataset("lex_glue", "case_hold")
    model = DistilBertFineTune()
    tokenizer = DistilBertTokenizer.from_pretrained(
        model.ckpt, use_fast=True, truncate=True, max_length=512
    )

    def preprocess_fn(examples):
        """
        Produces a collection of five token sequences, one per alternative, with the tokenizer
        merging the contexts and the holdings as desired
        """
        contexts = [[context] * 5 for context in examples["context"]]

        contexts = sum(contexts, [])

        holdings = sum(examples["endings"], [])

        tokenized_examples = tokenizer(contexts, holdings, truncation=True,max_length=512)
        features = {
            k: [v[i: i + 5] for i in range(0, len(v), 5)]
            for k, v in tokenized_examples.items()
        }
        return features

    tokenized_case_hold = case_hold.map(preprocess_fn, batched=True)
    tokenized_case_hold.set_format("torch")

    collator = DataCollatorForMultipleChoice(tokenizer)
    train_dataloader = DataLoader(
        tokenized_case_hold["train"], batch_size=16, collate_fn=collator,num_workers=7,persistent_workers=True
    )
    val_dataloader = DataLoader(
        tokenized_case_hold["validation"], batch_size=16, collate_fn=collator,num_workers=7,persistent_workers=True
    )

    wandb_logger:Logger = WandbLogger(log_model="all",project="case_hold_wrong_answers")


    trainer = Trainer(accelerator=hparams.accelerator,
                      logger=wandb_logger,
                      devices=hparams.devices,
                      val_check_interval=0.25, # large training set, check four times per epoch
                      max_epochs=hparams.epochs)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--epochs",default=2,type=int)
    args = parser.parse_args()
    main(args)


