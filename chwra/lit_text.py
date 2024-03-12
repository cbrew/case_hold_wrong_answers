"""
Multiple choice for case hold using lightning.
"""
from argparse import ArgumentParser


from pytorch_lightning import LightningModule, Trainer
from transformers import (
    DistilBertTokenizer,
    DistilBertModel,
)
from torch.utils.data.dataloader import DataLoader
import datasets
import torch
from torch import nn
from lightning.pytorch.loggers import WandbLogger, Logger
import torchmetrics

from chwra.collators import DataCollatorForMultipleChoice


class MultipleChoiceLightning(nn.Module):
    """
    Mirrors DistilBertForMultipleChoice.
    """

    def __init__(self, ckpt: str = "distilbert-base-uncased", wrong_answers=False):
        super().__init__()
        self.dim = 768  # think this is right for distilbery
        self.ckpt = ckpt
        self.distilbert = DistilBertModel.from_pretrained(self.ckpt)
        self.pre_classifier = nn.Linear(self.dim, self.dim)
        self.classifier = nn.Linear(self.dim, 1)
        self.dropout = nn.Dropout(p=0.1)
        self.wrong_answers: bool = wrong_answers

    def forward(self, input_ids, attention_mask):
        """
        forward pass of the main
        :param input_ids:
        :param attention_mask:
        :return:
        """
        num_choices = input_ids.shape[1]
        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        outputs = self.distilbert(input_ids, attention_mask=attention_mask)
        hidden_state = outputs[0]
        pooled_output = hidden_state[:, 0]
        pooled_output = self.pre_classifier(pooled_output)  # (bs * num_choices, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs * num_choices, dim)
        pooled_output = self.dropout(pooled_output)  # (bs * num_choices, dim)
        logits = self.classifier(pooled_output)  # (bs * num_choices, 1)

        if self.wrong_answers:
            return logits

        reshaped_logits = logits.view(-1, num_choices)  # (bs, num_choices)
        return reshaped_logits


class DistilBertFineTune(LightningModule):
    """ "
    Fine tuning module for distilbert multiple choice
    """

    def __init__(self, ckpt: str, wrong_answers: bool = False) -> None:
        super().__init__()
        self.distilbert = MultipleChoiceLightning(
            ckpt=ckpt, wrong_answers=wrong_answers
        )
        self.ckpt = ckpt
        self.save_hyperparameters()

        self.num_choices = 5
        self.wrong_answers = wrong_answers
        if self.wrong_answers:
            self.train_accuracy = torchmetrics.classification.Accuracy(
                task="multiclass", num_classes=self.num_choices
            )
            self.val_accuracy = torchmetrics.classification.Accuracy(
                task="multiclass", num_classes=self.num_choices
            )
        else:
            self.train_accuracy = torchmetrics.classification.Accuracy(task="binary")
            self.val_accuracy = torchmetrics.classification.Accuracy(task="binary")

    def training_step(self, *argmts,**kwargs):
        batch = argmts[0]
        preds, loss, labels = self.get_preds_loss_labels(batch)
        self.train_accuracy(preds, labels)
        self.log("train_accuracy", self.train_accuracy, on_epoch=True)
        self.log("train_loss", loss,on_epoch=True)
        return loss

    def validation_step(self, *argmts,**kwargs):
        batch = argmts[0]
        preds, loss, labels = self.get_preds_loss_labels(batch)
        self.val_accuracy(preds, labels)
        self.log("eval_loss", loss, on_epoch=True)
        self.log("eval_accuracy", self.val_accuracy, on_epoch=True)

    def get_preds_loss_labels(self, batch):
        """
        Shared function for training validation and testing.
        :param batch:
        :return:
        """
        labels = batch["labels"]
        logits = self.distilbert(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        if self.wrong_answers:
            # logits will be a flattened tensor with bs*num_choices elements in them.
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels.view(-1, 1))
            preds = logits.softmax(dim=1)
        else:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            preds = logits.argmax(dim=1)
        return preds, loss, labels

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-6)
        return optimizer


def main(hparams):
    """Main function as suggested in documentation for Trainer"""
    # recommended incantation to make good use of tensor cores.
    torch.set_float32_matmul_precision("medium")
    case_hold = datasets.load_dataset("lex_glue", "case_hold")
    model = DistilBertFineTune(
        ckpt=hparams.checkpoint, wrong_answers=hparams.wrong_answers
    )
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

        tokenized_examples = tokenizer(
            contexts, holdings, truncation=True, max_length=512
        )
        features = {
            k: [v[i : i + 5] for i in range(0, len(v), 5)]
            for k, v in tokenized_examples.items()
        }
        return features

    tokenized_case_hold = case_hold.map(preprocess_fn, batched=True)
    tokenized_case_hold.set_format("torch")

    collator = DataCollatorForMultipleChoice(tokenizer)
    train_dataloader = DataLoader(
        tokenized_case_hold["train"],
        batch_size=16,
        collate_fn=collator,
        num_workers=7,
        persistent_workers=True,
    )
    val_dataloader = DataLoader(
        tokenized_case_hold["validation"],
        batch_size=16,
        collate_fn=collator,
        num_workers=7,
        persistent_workers=True,
    )

    wandb_logger: Logger = WandbLogger(
        log_model="all", project="case_hold_wrong_answers"
    )
    wandb_logger.experiment.config.update({"model": hparams.checkpoint,
                                           "wrong_answers": hparams.wrong_answers,
                                           "max_epochs": hparams.epochs})

    trainer = Trainer(
        accelerator=hparams.accelerator,
        logger=wandb_logger,
        devices=hparams.devices,
        val_check_interval=0.25,  # large training set, check four times per epoch
        max_epochs=hparams.epochs,
    )
    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )


if __name__ == "__main__":
    eligible_distilberts = [
        "distilbert/distilbert-base-cased",
        "distilbert/distilbert-base-uncased",
    ]
    parser = ArgumentParser()
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--epochs", default=2, type=int)
    parser.add_argument("--wrong_answers", action="store_true")
    parser.add_argument(
        "--checkpoint",
        type=str,
        choices=eligible_distilberts,
        default="distilbert/distilbert-base-cased",
    )
    args = parser.parse_args()
    main(args)
