from transformers import AutoTokenizer, DistilBertForMultipleChoice
import torch
from torch.nn import  BCEWithLogitsLoss,CrossEntropyLoss,MSELoss

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
model = DistilBertForMultipleChoice.from_pretrained("distilbert-base-cased")

prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
choice0 = "It is eaten with a fork and a knife."
choice1 = "It is eaten while held in the hand."
choice2 = "It is eaten with a steam hammer."
choice3 = "Italians would not eat that."
labels = torch.tensor(0).unsqueeze(0)  # choice0 is correct (according to Wikipedia ;)), batch size 1

encoding = tokenizer([[prompt, choice0], [prompt, choice1],[prompt,choice2],[prompt,choice3]], return_tensors="pt", padding=True)
outputs = model(**{k: v.unsqueeze(0) for k, v in encoding.items()}, labels=labels)  # batch size is 1

# the linear classifier still needs to be trained
loss = outputs.loss
logits = outputs.logits
xhat = torch.argmax(logits, dim=1)

print(logits)
ce_lossfcn = CrossEntropyLoss()
loss = ce_lossfcn(logits, labels)
print(loss)

with torch.no_grad():
    bce_lossfcn = BCEWithLogitsLoss()
    other_labels = torch.tensor([[0,0,0,0]],dtype=torch.float32)

    print(bce_lossfcn(logits, other_labels))

    pred = logits.sigmoid() > 0.5
    print(pred)




