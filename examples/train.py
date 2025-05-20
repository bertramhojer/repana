from repana import Dataset, ReadingVector
import os
import pickle
import polars as pl

# load the example dataset
data = pl.read_csv("examples/data.csv")
examples = [f"{x} {y}" for x, y in zip(data["question"], data['answer'])]
dataset = Dataset(positive=examples)

# create the control vector
cv = ReadingVector(model_name="EleutherAI/pythia-14m", standardize=True)
# train and save the control vector
cv.train(dataset)
cv.save(complete_path="examples/control_vector.pkl")