from repana import ControlModel, ControlVector, ReadingContrastVector, ReadingVector, Dataset

def main():

    # test_data = []
    # with open("src/examples/data/ioi_test.txt", "r") as file:
    #     for line in file:
    #         test_data.append(line.strip().split("\t"))
    
    # X, y = zip(*test_data)
    
    
    model_name = "EleutherAI/pythia-410m"
    cv = ControlVector.load("src/examples/control-vectors/pythia-410m.json")

    alpha = None
    model = ControlModel(model_name=model_name, layer_ids=[12])

    results, accuracy = model.evaluate(cv=cv, alpha=alpha, X=["Then, Bob and Jamie went to the garden. Jamie gave a necklace to"], y=["Bob"], max_new_tokens=1)

    for res in results:
        print(res)
    print("Accuracy: ", accuracy)


if __name__ == "__main__":
    main()
