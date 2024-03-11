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
)
from torch.utils.data.dataloader import DataLoader
import datasets
import torch
from lightning.pytorch.loggers import WandbLogger

from chwra.collators import DataCollatorForMultipleChoice
import wandb

class DistilBertFineTune(LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.ckpt = "distilbert-base-uncased"
        self.distilbert: DistilBertForMultipleChoice = (
            DistilBertForMultipleChoice.from_pretrained(self.ckpt)
        )

    def training_step(self, batch, batch_idx):
        labels = batch["labels"]
        outputs = self.distilbert(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=labels,
        )
        # behind the scenes we get a torch.nn.CrossEntropyLoss
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        labels = batch["labels"]
        outputs = self.distilbert(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=labels,
        )
        self.log("val_loss", outputs.loss)
        # accuracy

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-6)
        return optimizer




def main(hparams):
    """Main function as suggeested in documentation for Trainer"""
    # recommended incantation to make good use of tensor cores.

    wandb.login()
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

        tokenized_examples = tokenizer(contexts, holdings, truncation=True)
        features = {
            k: [v[i: i + 5] for i in range(0, len(v), 5)]
            for k, v in tokenized_examples.items()
        }
        return features

    tokenized_case_hold = case_hold.map(preprocess_fn, batched=True)
    tokenized_case_hold.set_format("torch")

    collator = DataCollatorForMultipleChoice(tokenizer)
    train_dataloader = DataLoader(
        tokenized_case_hold["train"], batch_size=4, collate_fn=collator,num_workers=7,persistent_workers=True
    )
    val_dataloader = DataLoader(
        tokenized_case_hold["validation"], batch_size=4, collate_fn=collator,num_workers=7,persistent_workers=True
    )

    wandb_logger = WandbLogger(log_model="all",project="case_hold_wrong_answers")
    trainer = Trainer(logger=wandb_logger)
    trainer = Trainer(accelerator=hparams.accelerator,
                      devices=hparams.devices,
                      val_check_interval=0.10, # large training set, check ten times per epoch
                      max_epochs=hparams.epochs)
    trainer.fit(model, train_dataloader, val_dataloader)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--epochs",default=2,type=int)
    args = parser.parse_args()
    main(args)


