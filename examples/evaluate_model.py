from repana.controlModel import ControlModel
from repana.controlVector import ReadingVector
from repana.utils import evaluate, Dataset

model_name = "EleutherAI/Pythia-14m"
layer_ids = [5]
model = ControlModel(model_name, layer_ids)
dataset = Dataset(["The cat sat on the mat."], ["cat"])
cv = ReadingVector(model_name, standardize=False)
cv.train(dataset)

results_df, accuracy = evaluate(
    model_type="pythia",
    model=model,
    control_vector=cv,
    alpha=1.0,
    normalize=False,
    X=dataset.positive,
    y=["cat"],
    type="exact_match",
    batch_size=1
)
print(results_df)
print("Accuracy:", accuracy)
