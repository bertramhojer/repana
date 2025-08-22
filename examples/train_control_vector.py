from repana.controlVector import ReadingVector
from repana.utils import Dataset

# Replace with your actual model name and data
model_name = "gpt2"
dataset = Dataset(["The cat sat on the mat."], ["The dog barked."])

cv = ReadingVector(model_name, standardize=False)
cv.train(dataset)
cv.save("sample_task", "reading", shots=1)
print("Control vector trained and saved.")
