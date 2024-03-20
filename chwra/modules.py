import torch
import transformers
from torch import nn


class LinearCombinationLayer(nn.Module):
    """
    Trainable linear combination.
    """

    def __init__(self, init_value=0.5):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input1, input2):
        p = torch.sigmoid(self.weight)
        return p * input1 + (1 - p) * input2


class MultipleChoiceLightning(nn.Module):
    """
    module supporting multiple choice for case hold using lightning
    """

    def __init__(self,model,dropout=0.1):
        super().__init__()
        if isinstance(model, transformers.DistilBertModel):
            self.dim = 768
            self.model = model
            self.pre_classifier = nn.Linear(self.dim, self.dim)
            self.classifier = nn.Linear(self.dim, 1)
            self.dropout = nn.Dropout(p=dropout)
        elif isinstance(model, (transformers.RobertaModel, transformers.MPNetModel)):
            config = model.config
            self.dim = config.hidden_size
            self.model = model
            self.classifier = nn.Linear(self.dim, 1)
            self.dropout = nn.Dropout(p=dropout)
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
        num_choices = input_ids.shape[1]
        if isinstance(self.model, (transformers.RobertaModel, transformers.MPNetModel)):
            logits = self.forward_for_roberta(
                input_ids=input_ids, attention_mask=attention_mask
            )
        elif isinstance(self.model, transformers.DistilBertModel):
            logits = self.forward_for_distilbert(
                input_ids=input_ids, attention_mask=attention_mask
            )
        else:
            raise ValueError(f"Unsupported model {self.model}")
        reshaped_logits = logits.view(-1, num_choices)
        return reshaped_logits  #  (bs,num_choices)

    def forward_for_distilbert(self, input_ids, attention_mask):
        """
        Forward pass if the model is a distilbert.
        :param input_ids:
        :param attention_mask:
        :return:
        """

        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        outputs = self.model(input_ids, attention_mask=attention_mask)
        hidden_state = outputs[0]
        pooled_output = hidden_state[:, 0]
        pooled_output = self.pre_classifier(pooled_output)  # (bs * num_choices, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs * num_choices, dim)
        pooled_output = self.dropout(pooled_output)  # (bs * num_choices, dim)
        logits = self.classifier(pooled_output)  # (bs * num_choices, 1)
        return logits

    def forward_for_roberta(self, input_ids, attention_mask):
        """
        forward pass if the model is roberta. Very similar to what is done for distilbert
        above, except there is no pre-classifier layer in the model.

        :param input_ids:
        :param attention_mask:
        :return:
        """
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        outputs = self.model(
            flat_input_ids,
            attention_mask=flat_attention_mask,
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class EnsembleClassifier(nn.Module):
    def __init__(self, ckpt: str = "distilbert-base-uncased"):
        super().__init__()
        self.ckpt = ckpt
        self.ra_classifier = MultipleChoiceLightning(ckpt)
        self.wa_classifier = MultipleChoiceLightning(ckpt)
        self.linear_combination = LinearCombinationLayer()

    def forward(self, input_ids, attention_mask):
        logits1 = self.ra_classifier(input_ids, attention_mask)
        logits2 = self.wa_classifier(input_ids, attention_mask)
        return self.linear_combination(logits1, logits2)
