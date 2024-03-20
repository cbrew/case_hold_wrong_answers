"""
Multiple choice for case hold using lightning.

Alternate loss function for case hold.
"""
from argparse import ArgumentParser
import os

from dotenv import load_dotenv
import lightning
from lightning import LightningModule, Trainer
from lightning.pytorch.loggers import WandbLogger, Logger
from lightning.pytorch.callbacks import EarlyStopping,ModelCheckpoint
from torch.utils.data.dataloader import DataLoader
import datasets
import torch
from torch import nn
from torch.nn.functional import one_hot


import torchmetrics
import transformers
from chwra.collators import DataCollatorForMultipleChoice
from chwra.modules import MultipleChoiceLightning


class DistilBertFineTune(LightningModule):
    """ "
    Fine tuning module for multiple choice.
    """

    def __init__(
        self,
        model,
        model_name,
        wrong_answers = False,
        right_answers = True,
        learning_rate = 1e-6,
        dropout=0.1,
        **kwargs) -> None:
        super().__init__()
        self.mul_module = MultipleChoiceLightning(model=model,dropout=dropout)
        self.ckpt = model_name
        self.save_hyperparameters({})
        self.learning_rate =  learning_rate
        self.num_choices = 5
        self.wrong_answers = wrong_answers
        self.right_answers = right_answers
        if not self.right_answers or self.wrong_answers:
            self.right_answers = False

        self.train_accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=self.num_choices
        )
        self.train_f1 = torchmetrics.classification.F1Score(
            task="multiclass", average="micro", num_classes=self.num_choices
        )

        self.train_precision = torchmetrics.classification.Precision(
            task="multiclass", average="micro", num_classes=self.num_choices
        )

        self.train_recall = torchmetrics.classification.Recall(
            task="multiclass", average="micro", num_classes=self.num_choices
        )

        self.val_accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=self.num_choices
        )

        self.val_f1 = torchmetrics.classification.F1Score(
            task="multiclass", average="micro", num_classes=self.num_choices
        )

        self.val_precision = torchmetrics.classification.Precision(
            task="multiclass", average="micro", num_classes=self.num_choices
        )

        self.val_recall = torchmetrics.classification.Recall(
            task="multiclass", average="micro", num_classes=self.num_choices
        )
        self.save_hyperparameters(ignore=['model'])

    def loss_calc(self, logits, labels, preds):
        if self.wrong_answers and self.right_answers:
            loss = self.get_loss_wa(logits, labels) * 0.10 + self.get_loss_ra(preds, labels) * 0.9
        elif self.wrong_answers:
            loss = self.get_loss_wa(logits, labels)
        elif self.right_answers:
            loss = self.get_loss_ra(logits, labels)
        else:
            loss = self.get_loss_ra(logits, labels)
        return loss
    def training_step(self, *argmts, **kwargs):
        batch = argmts[0]
        labels = batch["labels"]  # (bs,num_choices)
        logits = self.mul_module(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        preds = logits.argmax(dim=-1)
        loss = self.loss_calc(logits, labels, preds)
        self.train_accuracy(preds, labels)
        self.train_f1(preds, labels)
        self.train_precision(preds, labels)
        self.train_recall(preds, labels)
        self.log("train_accuracy", self.train_accuracy)
        self.log("train_f1", self.train_f1)
        self.log("train_loss", loss)
        self.log("train_precision", self.train_precision)
        self.log("train_recall", self.train_recall)
        return loss




    def validation_step(self, *argmts, **kwargs):
        batch = argmts[0]
        labels = batch["labels"]  # (bs,num_choices)
        logits = self.mul_module(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        preds = logits.argmax(dim=-1)
        with torch.no_grad():
            loss = self.loss_calc(logits,labels,preds)

        self.val_accuracy(preds, labels)
        self.val_f1(preds, labels)
        self.val_precision(preds, labels)
        self.val_recall(preds, labels)
        self.log("eval_loss", loss)
        self.log("eval_accuracy", self.val_accuracy)
        self.log("eval_f1", self.val_f1)
        self.log("eval_precision", self.val_precision)
        self.log("eval_recall", self.val_recall)

    def get_loss_wa(self, logits, labels):
        """
        get loss from wrong answers
        """

        loss_fn = nn.BCEWithLogitsLoss()
        hot_labels = one_hot(labels, num_classes=5).float()  # (bs,num_choices)
        return loss_fn(logits, hot_labels)

    def get_loss_ra(self, logits, labels):
        """
        Run the model and get loss and predictions from right answers
        """
        loss_fn = nn.CrossEntropyLoss()
        return loss_fn(logits, labels)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.06,

        )
        return optimizer


def main(checkpoint=None,
         seed=42,
         learning_rate=1e-6,
         batch_size=8,
         accumulate_grad_batches=4,
         right_answers=True,
         wrong_answers=False,
         epochs=1,
         dropout=0.1,
         accelerator="auto",):
    """Main function as suggested in documentation for Trainer"""
    # recommended incantation to make good use of tensor cores.
    lightning.seed_everything(seed)
    torch.set_float32_matmul_precision("medium")
    case_hold = datasets.load_dataset("coastalcph/lex_glue", "case_hold")
    base_model = transformers.AutoModel.from_pretrained(checkpoint)
    model = DistilBertFineTune(base_model,model_name=checkpoint,learning_rate=learning_rate,dropout=dropout)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model.ckpt,
        use_fast=True,
        truncate="only_first",
        max_length=512,  # ??? truncation handling
    )

    def preprocess_fn(examples):
        """
        Produces a collection of five token sequences, one per alternative, with the tokenizer
        merging the contexts and the holdings as desired
        """

        holdings = []
        for context,endings in zip(examples["context"],examples["endings"]):
            for ending in endings:
                holdings.append(context.replace("<HOLDING>",ending))

        tokenized_examples = tokenizer(
            holdings, truncation=True, max_length=512
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
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=7,
        persistent_workers=True,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        tokenized_case_hold["validation"],
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=7,
        persistent_workers=True,
    )

    wandb_logger: Logger = WandbLogger(
        log_model="all", project="case_hold_wrong_answers"
    )
    assert isinstance(wandb_logger, WandbLogger)
    wandb_logger.experiment.config.update(
        {
            "wrong_answers": wrong_answers,
            "right_answers": right_answers,
            "max_epochs": epochs,
        }
    )
    logger: Logger = wandb_logger
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="eval_f1",
        dirpath="experiments",
        filename="case-hold-{epoch:02d}-{eval_f1:.4f}"
    )
    early_stopping = EarlyStopping('eval_f1', patience=3,mode="max")
    trainer = Trainer(
        accelerator=accelerator,
        logger=logger,
        accumulate_grad_batches=accumulate_grad_batches,
        precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
        val_check_interval=0.20,
        max_epochs=epochs,
        callbacks=[checkpoint_callback,early_stopping],
    )
    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    load_dotenv()
    eligible_models = [
        "distilbert/distilbert-base-cased",
        "distilbert/distilbert-base-uncased",
        "distilbert/distilroberta-base",
        "FacebookAI/roberta-base",
        "FacebookAI/roberta-large",
        "sentence-transformers/all-mpnet-base-v2",
        "lexlms/legal-roberta-base",
        "lexlms/legal-roberta-large",
    ]
    parser = ArgumentParser()
    parser.add_argument("--seed",type=int, default=42)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--epochs", default=4, type=int)
    parser.add_argument("--accumulate_grad_batches", default=8, type=int)
    parser.add_argument("--wrong_answers", action="store_true")
    parser.add_argument("--right_answers", action="store_true")
    parser.add_argument("--learning_rate", default=5e-5, type=float) # maybe too high
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument(
        "--checkpoint",
        type=str,
        choices=eligible_models,
        default="distilbert/distilbert-base-cased",
    )
    args = parser.parse_args()
    main(**vars(args))
