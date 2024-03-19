from lit_text import DistilBertFineTune


model = DistilBertFineTune.load_from_checkpoint("wrong_answers_best/model.ckpt")
print(model)