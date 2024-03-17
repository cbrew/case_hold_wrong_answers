#  Progress

lit_text.py does the same as the huggingface multiple choice distilbert, but more explicitly.

It appears that the wrong answer loss is different from the right answer score, but not clear that it is
more than a scaling issue.

selecting learning rates for Roberta large, trying 3.2e-5, which is close to 3e-5 as reported in Chalkidis
Also added early stopping, which makes it probable that < 1 epoch is desirable. Next experiment should probe this, by
saving checkpoints more often.



