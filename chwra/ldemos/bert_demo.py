from pytorch_lightning import LightningModule
from transformers import BertModel, BertTokenizer
from torch import nn


from datasets import load_dataset

dataset = load_dataset("nyu-mll/multi_nli")

class BertMNLIFinetuner(LightningModule):
    def __init__(self):
        super().__init__()

        self.bert = BertModel.from_pretrained("bert-base-cased", output_attentions=True)
        self.W = nn.Linear(self.bert.config.hidden_size, 3)
        self.num_classes = 3

    def forward(self, input_ids, attention_mask, token_type_ids):
        h, _, attn = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        h_cls = h[:, 0]
        logits = self.W(h_cls)
        return logits, attn


def main():
    dataset = load_dataset("nyu-mll/multi_nli")
    print(dataset)


if __name__ == "__main__":
    main()
