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
labels = torch.tensor(3).unsqueeze(0)  # choice3 is prefrred off the bat

encoding = tokenizer([[prompt, choice0], [prompt, choice1],[prompt,choice2],[prompt,choice3]], return_tensors="pt", padding=True)
outputs = model(**{k: v.unsqueeze(0) for k, v in encoding.items()}, labels=labels)  # batch size is 1

# the linear classifier still needs to be trained
loss1 = outputs.loss
logits = outputs.logits
print("CE loss from hg",loss1,logits.shape,labels)


ce_lossfcn = CrossEntropyLoss()
loss2 = ce_lossfcn(logits, labels)
print("CE loss",loss1,logits.shape,labels.shape)

with torch.no_grad():
    bce_lossfcn = BCEWithLogitsLoss()
    targets = torch.tensor([[1,0,0,1]],dtype=torch.float32)
    assert logits.shape == targets.shape
    print("BCE Loss",bce_lossfcn(logits, targets),logits.shape,targets.shape)






