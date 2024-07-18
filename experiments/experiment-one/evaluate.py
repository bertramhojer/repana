from repana import ControlModel, ControlVector, PCAContrastVector, ReadingContrastVector, ReadingVector, Dataset, evaluate
import pickle
import json
import os
from types import SimpleNamespace


with open('config.json', 'r') as config_file:
    config_data = json.load(config_file)
    config = SimpleNamespace(**config_data)


# Load results.json file
results_path = 'results.json'
if os.path.exists(results_path):
    with open(results_path, 'r') as results_file:
        results_data = json.load(results_file)
else:
    results_data = {}


def add_results(accuracy, results, alpha):

    model_name = config.model_name.split('/')[-1]
    if model_name not in results_data:
        results_data[model_name] = {}
    
    results_data[model_name][f"alpha_{alpha}"] = {
        "accuracy": accuracy,
        "results": results
    }

    # Save updated results to the JSON file
    with open(results_path, 'w') as results_file:
        json.dump(results_data, results_file, indent=4)


with open('config.json', 'r') as config_file:
    config_data = json.load(config_file)
    config = SimpleNamespace(**config_data)


def run():

    cv_path = f"control-vectors/{config.cv_type}/{config.model_name.split('/')[-1]}-{config.shots}.json"

    with open(f"data/{config.task}.pkl", "rb") as file:
        data = pickle.load(file)
    
    X, y = data[config.test_shots]["X_test"][:100], data[config.test_shots]["y_test"][:100]

    model = ControlModel(model_name=config.model_name, layer_ids=[1])
    cv = ControlVector.load(cv_path)

    settings = {
            "pad_token_id": model.tokenizer.eos_token_id,  # silence warning
            "do_sample": False,  # temperature=0
            "max_new_tokens": 1,
            "repetition_penalty": 1.1,  # reduce control jank
        }
    
    for a in config.alpha:
        results, accuracy = evaluate(model=model, control_vector=cv, normalize=config.normalize, alpha=a, X=X, y=y, settings=settings)
        add_results(accuracy, results, a)


if __name__=="__main__":
    run()