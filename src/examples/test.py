from repana import ControlModel, ControlVector, ReadingContrastVector, ReadingVector, Dataset, evaluate
import pickle
from typing import Literal

def main(state: Literal["train", "evaluate"] = "train"):

    if state == "train":

        with open("src/examples/data/ioi_train.pkl", "rb") as file:
            train_data = pickle.load(file)

        # Create a list of strings from the list of tuples
        train_data = [f"{x[0]} {x[1]}" for x in train_data]

        model_name = "EleutherAI/pythia-410m"
        dataset = Dataset(positive=train_data, negative=None)
        cv = ReadingVector(model_name=model_name)
        cv.train(dataset=dataset)
        cv.save("src/examples/control-vectors/pythia-410m.json")


    if state == "evaluate":
    
        # Load the test data from the pickle file
        with open("src/examples/data/ioi_test.pkl", "rb") as file:
            test_data = pickle.load(file)

        X, y = zip(*test_data)

        model_name = "EleutherAI/pythia-410m"
        cv = ControlVector.load("src/examples/control-vectors/pythia-410m.json")

        alpha = 0
        model = ControlModel(model_name=model_name, layer_ids=[20])

        settings = {
                "pad_token_id": model.tokenizer.eos_token_id, # silence warning
                "do_sample": False, # temperature=0
                "max_new_tokens": 1,
                "repetition_penalty": 1.1, # reduce control jank
            }

        evaluate(model=model, control_vector=cv, alpha=alpha, X=X, y=y, settings=settings)



if __name__ == "__main__":
    main(state="evaluate")
