from repana import ControlModel, ControlVector, ReadingContrastVector, ReadingVector, Dataset, evaluate
import pickle
from typing import Literal

def main(state: Literal["train", "evaluate"] = "train"):

    if state == "train":

        with open("src/examples/data/A.pkl", "rb") as file:
            train_data = pickle.load(file)

        # Create a list of strings from the list of tuples
        positive = [f"{x} {y}" for x, y in zip(train_data[1]["X_train"], train_data[1]["y_train"])]
        negative = train_data[1]["X_train_negative"]

        model_name = "EleutherAI/pythia-410m"
        dataset = Dataset(positive=positive, negative=negative)
        cv = ReadingContrastVector(model_name=model_name)
        cv.train(dataset=dataset)
        cv.save("src/examples/control-vectors/pythia-410m-A1.json")


    if state == "evaluate":
    
        # Load the test data from the pickle file
        with open("src/examples/data/ioi_test.pkl", "rb") as file:
            test_data = pickle.load(file)

        X, y = zip(*test_data)

        model_name = "EleutherAI/pythia-410m"
        model = ControlModel(model_name=model_name, layer_ids=[1])

        cv = ControlVector.load("src/examples/control-vectors/pythia-410m-A1.json")
        alpha = .5
        normalize = False

        settings = {
            "pad_token_id": model.tokenizer.eos_token_id,  # silence warning
            "do_sample": False,  # temperature=0
            "max_new_tokens": 1,
            "repetition_penalty": 1.1,  # reduce control jank
        }

        evaluate(model=model, control_vector=cv, normalize=normalize, alpha=alpha, X=X, y=y, settings=settings)



if __name__ == "__main__":
    main(state="evaluate")
