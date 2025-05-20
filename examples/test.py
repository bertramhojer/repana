from repana import Dataset, ControlVector, ControlModel, evaluate
import os
import pickle
import polars as pl

def get_model_settings(model: ControlModel):
    return {
        "pad_token_id": model.tokenizer.eos_token_id,
        "do_sample": False,
        "max_new_tokens": 10,
        "stop_strings": ["\n", "Passage:"],
        "tokenizer": model.tokenizer,
    }

# set variables
model_name = "EleutherAI/pythia-14m"
layer_id = 2
alpha = 10
eval_type = "logit"
batch_size = 32

# laod data
data = pl.read_csv("examples/data.csv")[:10]
X, y = list(data["question"]), list(data['answer'])
answer_list = list(set(y))

# instantiate model and load control vector
model = ControlModel(model_name=model_name, layer_ids=[layer_id])
settings = get_model_settings(model)
cv = ControlVector.load("examples/control_vector.pkl")

# evaluate
answers, accuracy = evaluate(
            model_type='pythia', model=model, control_vector=cv, alpha=alpha,
            normalize=False, X=X, y=y, type=eval_type, answer_list=answer_list,
            batch_size=batch_size, settings=settings
        )

print(f"Answers: {answers}")
print(f"Accuracy: {accuracy}")