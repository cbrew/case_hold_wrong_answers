import os
import wandb
import dotenv

dotenv.load_dotenv()
wandb.login(key=os.getenv("WANDB_API_KEY"))
run = wandb.init(project="case_hold_wrong_answers")
artifact = run.use_artifact("cbrew81/case_hold_wrong_answers/model-wgdmopqs:v3")
artifact.download("wrong_answers_best")