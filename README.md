# case_hold_wrong_answers

This is an adaptation of 

@inproceedings{Kim2020LearningTC,
  title={Learning to Classify the Wrong Answers for Multiple Choice Question Answering (Student Abstract)},
  author={Hyeon-Jin Kim and Pascale Fung},
  booktitle={AAAI Conference on Artificial Intelligence},
  year={2020},
  url={https://api.semanticscholar.org/CorpusID:219182195}
}

to the Case Hold task from lex_glue. 

The task is multiple choice, 

The idea from the paper is to run two models. One is trained to pick the right alternative, the
other to avoid the wrong alternative.

To do, current version is doing only one model, with a small model. Need to add the second model.


