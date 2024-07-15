from repana import ControlModel, ControlVector, ReadingContrastVector, Dataset

def main():

    model_name = "EleutherAI/pythia-14m"

    # cv = ReadingContrastVector(model_name=model_name)
    # dataset = Dataset(positive=["I am happy", "I am great", "All is okay"], negative=["I am sad", "I am bad", "It is shit"])
    # cv.train(dataset)
    # cv.save("src/examples/control-vectors/test.json")

    cv = ControlVector.load("src/examples/control-vectors/test.json")


    alpha = 5
    model = ControlModel(model_name=model_name, layer_ids=[3])
    model.set_control(control_vector=cv.directions, alpha=alpha)
    tokenizer = model.tokenizer

    input = f"I am happy"

    # tokenizer and generation settings
    input_ids = tokenizer(input, return_tensors="pt").to(model.device)
    settings = {
        "pad_token_id": tokenizer.eos_token_id, # silence warning
        "do_sample": False, # temperature=0
        "max_new_tokens": 128,
        "repetition_penalty": 1.1, # reduce control jank
    }
    
    print(tokenizer.decode(model.generate(**input_ids, **settings).squeeze()))


if __name__ == "__main__":
    main()
