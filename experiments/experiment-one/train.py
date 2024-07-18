from repana import ControlModel, ControlVector, PCAContrastVector, PCAReadingVector, ReadingContrastVector, ReadingVector, Dataset, evaluate
import pickle
import json
from types import SimpleNamespace


with open('config.json', 'r') as config_file:
    config_data = json.load(config_file)
    config = SimpleNamespace(**config_data)


def run():

    with open(f"data/{config.task}.pkl", "rb") as file:
        data = pickle.load(file)
    
    if config.cv_type == "R":
        cv = ReadingVector(model_name=config.model_name)
        positive = [f"{x} {y}" for x, y in zip(data[config.shots]["X_train"], data[config.shots]["y_train"])]
        negative = None
    else:
        positive = [f"{x} {y}" for x, y in zip(data[config.shots]["X_train"], data[config.shots]["y_train"])]
        negative = data[config.shots]["X_train_negative"]

        if config.cv_type == "R-C":
            cv = ReadingContrastVector(model_name=config.model_name)
        elif config.cv_type == "PCA-R":
            cv = PCAReadingVector(model_name=config.model_name)
        elif config.cv_type == "PCA-C":
            cv = PCAContrastVector(model_name=config.model_name)
    
    dataset = Dataset(positive=positive, negative=negative)
    cv.train(dataset)
    cv.save(f"control-vectors/{config.cv_type}/{config.model_name.split('/')[-1]}-{config.shots}.json")



if __name__=="__main__":
    run()