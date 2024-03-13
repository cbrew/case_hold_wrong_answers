"""
Multiple choice for case hold using lightning.
"""
from argparse import ArgumentParser
import os


import lightning
from lightning import LightningModule, Trainer
from lightning.pytorch.loggers import WandbLogger, Logger
from torch.utils.data.dataloader import DataLoader
import datasets
import torch
from torch import nn


import torchmetrics
import transformers

from chwra.collators import DataCollatorForMultipleChoice


class MultipleChoiceLightning(nn.Module):
    """
    module supporting multiple choice for case hold using lightning
    """

    def __init__(self, ckpt: str = "distilbert-base-uncased", wrong_answers=False):
        super().__init__()

        self.ckpt: str = ckpt
        model = transformers.AutoModel.from_pretrained(ckpt)


        self.wrong_answers: bool = wrong_answers

        if isinstance(model,transformers.DistilBertModel):
            self.dim = 768
            self.model = model
            self.pre_classifier = nn.Linear(self.dim, self.dim)
            self.classifier = nn.Linear(self.dim, 1)
            self.dropout = nn.Dropout(p=0.1)  # ??? dropout correct
        elif isinstance(model,(transformers.RobertaModel,transformers.MPNetModel)):
            config = model.config
            self.dim = config.hidden_size
            self.model = model
            self.classifier = nn.Linear(self.dim, 1)
            self.dropout = nn.Dropout(p=0.1)
        else:
            raise ValueError(f"Unsupported model {self.model}")

    def forward(self, input_ids, attention_mask):
        """
        forward pass of the model, mirrors how it is handled within the
        huggingface multiple choice model.
        :param input_ids:
        :param attention_mask:
        :return:
        """
        if isinstance(self.model, (transformers.RobertaModel, transformers.MPNetModel)):
            return self.forward_for_roberta(
                input_ids=input_ids, attention_mask=attention_mask
            )
        elif isinstance(self.model, transformers.DistilBertModel):
            return self.forward_for_distilbert(
                input_ids=input_ids, attention_mask=attention_mask
            )

    def forward_for_distilbert(self, input_ids, attention_mask):
        """
        Forward pass if the model is a distilbert.
        :param input_ids:
        :param attention_mask:
        :return:
        """
        num_choices = input_ids.shape[1]
        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        outputs = self.model(input_ids, attention_mask=attention_mask)
        hidden_state = outputs[0]
        pooled_output = hidden_state[:, 0]
        pooled_output = self.pre_classifier(pooled_output)  # (bs * num_choices, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs * num_choices, dim)
        pooled_output = self.dropout(pooled_output)  # (bs * num_choices, dim)
        logits = self.classifier(pooled_output)  # (bs * num_choices, 1)
        reshaped_logits = logits.view(-1, num_choices)  # (bs, num_choices)
        return reshaped_logits

    def forward_for_roberta(self, input_ids, attention_mask):
        """
        forward pass if the model is roberta. Very similar to what is done for distilbert
        above, except there is no pre-classifier layer in the model.

        :param input_ids:
        :param attention_mask:
        :return:
        """

        num_choices = input_ids.shape[1]
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        outputs = self.model(
            flat_input_ids,
            attention_mask=flat_attention_mask,
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)
        return reshaped_logits


class DistilBertFineTune(LightningModule):
    """ "
    Fine tuning module for distilbert multiple choice
    """

    def __init__(
        self, ckpt: str, learning_rate: float, wrong_answers: bool = False
    ) -> None:
        super().__init__()
        self.distilbert = MultipleChoiceLightning(
            ckpt=ckpt, wrong_answers=wrong_answers
        )
        self.ckpt = ckpt
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.num_choices = 5
        self.wrong_answers = wrong_answers
        if self.wrong_answers:
            self.train_accuracy = torchmetrics.classification.Accuracy(task="binary")
            self.train_f1 = torchmetrics.classification.F1Score(task="binary")
            self.val_accuracy = torchmetrics.classification.Accuracy(task="binary")
            self.val_f1 = torchmetrics.classification.F1Score(task="binary")
        else:
            self.train_accuracy = torchmetrics.classification.Accuracy(
                task="multiclass", num_classes=self.num_choices
            )
            self.train_f1 = torchmetrics.classification.F1Score(
                task="multiclass", average="micro", num_classes=self.num_choices
            )

            self.val_accuracy = torchmetrics.classification.Accuracy(
                task="multiclass", num_classes=self.num_choices
            )

            self.val_f1 = torchmetrics.classification.F1Score(
                task="multiclass", average="micro", num_classes=self.num_choices
            )

    def training_step(self, *argmts, **kwargs):
        batch = argmts[0]
        preds, loss, labels = self.get_preds_loss_labels(batch)
        self.train_accuracy(preds, labels)
        self.train_f1(preds, labels)
        self.log("train_accuracy", self.train_accuracy)
        self.log("train_f1", self.train_f1)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, *argmts, **kwargs):
        batch = argmts[0]
        preds, loss, labels = self.get_preds_loss_labels(batch)
        self.val_accuracy(preds, labels)
        self.val_f1(preds, labels)
        self.log("eval_loss", loss)
        self.log("eval_accuracy", self.val_accuracy)
        self.log("eval_f1", self.val_f1)

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
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(4.0))  # ??? weight
            # reasoning here is to make the positive class "not wrong answer"
            # but still require that wrong answers be driven towards zero.
            labels = nn.functional.one_hot(labels, num_classes=5).float()
            loss = loss_fn(logits, labels)
            preds = (logits.sigmoid() > 0.5).float()  # ??? threshold

        else:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            preds = logits.argmax(dim=1)
        return preds, loss, labels

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=1e-6
        )  # ??? good learning rate
        return optimizer


def main(hparams):
    """Main function as suggested in documentation for Trainer"""
    # recommended incantation to make good use of tensor cores.
    torch.set_float32_matmul_precision("medium")
    case_hold = datasets.load_dataset("lex_glue", "case_hold")
    model = DistilBertFineTune(
        learning_rate=hparams.learning_rate,
        ckpt=hparams.checkpoint,
        wrong_answers=hparams.wrong_answers,
    )
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
        contexts = [[context] * 5 for context in examples["context"]]

        contexts = sum(contexts, [])

        holdings = sum(examples["endings"], [])

        tokenized_examples = tokenizer(
            contexts, holdings, truncation="only_first", max_length=512
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
        batch_size=8,
        collate_fn=collator,
        num_workers=7,
        persistent_workers=True,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        tokenized_case_hold["validation"],
        batch_size=8,
        collate_fn=collator,
        num_workers=7,
        persistent_workers=True,
    )

    wandb_logger: Logger = WandbLogger(
        log_model="all", project="case_hold_wrong_answers"
    )
    assert isinstance(wandb_logger, WandbLogger)
    wandb_logger.experiment.config.update(
        {"wrong_answers": hparams.wrong_answers, "max_epochs": hparams.epochs}
    )
    logger: Logger = wandb_logger
    checkpoint_callback = lightning.pytorch.callbacks.ModelCheckpoint(
        save_top_k=2, monitor="eval_f1"
    )
    trainer = Trainer(
        accelerator=hparams.accelerator,
        logger=logger,
        devices=hparams.devices,
        precision="bf16" if torch.cuda.is_available() else "32-true",
        val_check_interval=0.5,
        max_epochs=hparams.epochs,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    lightning.seed_everything(42)
    eligible_models = [
        "distilbert/distilbert-base-cased",
        "distilbert/distilbert-base-uncased",
        "distilbert/distilroberta-base",
        "FacebookAI/roberta-base",
        "sentence-transformers/all-mpnet-base-v2",
    ]
    parser = ArgumentParser()
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--epochs", default=2, type=int)
    parser.add_argument("--wrong_answers", action="store_true")
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument(
        "--checkpoint",
        type=str,
        choices=eligible_models,
        default="distilbert/distilbert-base-cased",
    )
    args = parser.parse_args()
    main(args)
