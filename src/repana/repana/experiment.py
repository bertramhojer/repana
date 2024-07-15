from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from myrep import ControlVector, ControlModel, DatasetEntry
import dataclasses
from typing import List
import torch
import json
import pickle
import os
import numpy as np
import tqdm
from abc import ABC, abstractmethod
from sklearn.decomposition import PCA
import warnings


class Experiment:

    def __init__(self, dataset, control_vectors=None, layers=None, scales=None):
        self.dataset = dataset
        self.control_vectors = control_vectors


    def run(self, model, control_vector, dataset):
        pass

    def eval(self):
        pass



class ControlModel(torch.nn.Module):

    def __init__(self, model_name, layer_ids):
        super().__init__()
        self.model_name = model_name
        self.layer_ids = layer_ids
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.layers = model_layer_list(self.model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token_id = 0

        self._create_control_model()
    

    def _create_control_model(self):

        for layer_id in self.layer_ids:
            layer = self.layers[layer_id]
            if not isinstance(layer, ControlBlock):
                self.layers[layer_id] = ControlBlock(layer)
    

    def set_control(self, control_vector, alpha):

        for layer_id in self.layer_ids:
            self.layers[layer_id].control_vector = control_vector[layer_id]
            self.layers[layer_id].alpha = alpha
    

    @property
    def device(self) -> torch.device:
        return self.model.device

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)
            



class ControlBlock(torch.nn.Module):

    control_vector = None
    alpha = float(1.0)

    def __init__(self, layer):
        
        super().__init__()
        self.block = layer

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)

        if self.control_vector is None:
            return output
        
        if isinstance(output, tuple):
            modified = output[0]
        else:
            modified = output
        
        assert modified.shape[-1] == self.control_vector.shape[0], f"OUT SHAPE: {modified.shape}, CV SHAPE: {self.control_vector.shape}"
        
        modified += self.control_vector * self.alpha

        if isinstance(output, tuple):
            output = (modified,) + output[1:]
        else:
            output = modified
    
        return output




@dataclasses.dataclass
class ControlVector(ABC):
    model_name: str
    directions: dict[int, np.ndarray] = dataclasses.field(default_factory=dict)


    @abstractmethod
    def train(self, dataset):
        pass
    

    def _read_representations(self, dataset, batch_size=32):
        """
        Read representations from the model for the given dataset.
        Checks whether there are positive and negative examples in the dataset.
        If there are, it reads the representations for the positive and negative examples and returns them.
        If there are no negative examples, it returns the representations for the positive examples and negative as {}.
        """

        model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.pad_token_id = 0

        model_layers = model_layer_list(model)
        self.n_layers = len(model_layers)

        positive_representations: dict[int, np.ndarray] = {}
        negative_representations: dict[int, np.ndarray] = {}

        positive_batches = [
            dataset.positive[p : p + batch_size] for p in range(0, len(dataset.positive), batch_size)
        ]

        hidden_states_positives = {layer: [] for layer in range(self.n_layers)}
        with torch.no_grad():
            for batch in tqdm.tqdm(positive_batches):
                out = model(
                    **tokenizer(batch, padding=True, return_tensors="pt").to(model.device),
                    output_hidden_states=True,
                )
                out = out.hidden_states[-self.n_layers:]
                for i in range(self.n_layers):
                    hidden_states_positives[i].append(out[i][:, -1, :].cpu().numpy())
                del out
            
            positive_representations = {l: np.vstack(hidden_states_positives[l]) for l in range(self.n_layers)}

            if dataset.negative != None:

                negative_batches = [
                    dataset.negative[p : p + batch_size] for p in range(0, len(dataset.negative), batch_size)
                ]

                hidden_states_negatives = {layer: [] for layer in range(self.n_layers)}
                with torch.no_grad():
                    for batch in tqdm.tqdm(negative_batches):
                        out = model(
                            **tokenizer(batch, padding=True, return_tensors="pt").to(model.device),
                            output_hidden_states=True,
                        )
                        out = out.hidden_states[-self.n_layers:]
                        for i in range(self.n_layers):
                            hidden_states_negatives[i].append(out[i][:, -1, :].cpu().numpy())
                        del out
                    
                    negative_representations = {l: np.vstack(hidden_states_negatives[l]) for l in range(self.n_layers)}
            
        return positive_representations, negative_representations
    

    def _save_control_vector(self):
        pass



@dataclasses.dataclass
class ReadingVector(ControlVector):
    additional_param: float = 1.0

    def train(self, dataset):

        positive_representations, _ = self._read_representations(dataset)

        for layer in tqdm.tqdm(range(self.n_layers)):

            h = positive_representations[layer]
            self.directions[layer] = np.mean(h, axis=0)
        
        #print("##### ", self.directions)
        
        return self.directions


    def additional_method(self):
        # Add any additional methods specific to the derived class
        pass



@dataclasses.dataclass
class ReadingContrastVector(ControlVector):
    additional_param: float = 1.0

    def train(self, dataset):

        positive_representations, negative_representations = self._read_representations(dataset)

        for layer in tqdm.tqdm(range(self.n_layers)):

            h_positive = positive_representations[layer]
            h_negative = negative_representations[layer]

            h = h_positive - h_negative

            self.directions[layer] = np.mean(h, axis=0)
    
        return self



@dataclasses.dataclass
class PCAContrastVector(ControlVector):
    additional_param: float = 1.0

    def train(self, dataset):

        positive_representations, negative_representations = self._read_representations(dataset)

        for layer in tqdm.tqdm(range(self.n_layers)):

            h_positive = positive_representations[layer]
            h_negative = negative_representations[layer]

            h = h_positive - h_negative

            pca_model = PCA(n_components=1).fit(h)
            self.directions[layer] = pca_model.components_.astype(np.float32).squeeze(axis=0)
        
        print("#### ", self.directions)

        return self.directions


    def additional_method(self):
        # Add any additional methods specific to the derived class
        pass





@dataclasses.dataclass
class Dataset:

    def __init__(self, positive, negative=None):
        if negative is not None:
            assert len(positive) == len(negative), "Positive and negative datasets must have the same length"
        self.positive = positive
        self.negative = negative


def model_layer_list(model: ControlModel | PreTrainedModel) -> torch.nn.ModuleList:
    if isinstance(model, ControlModel):
        model = model.model

    if hasattr(model, "model"):  # mistral-like
        return model.model.layers
    elif hasattr(model, "transformer"):  # gpt-2-like
        return model.transformer.h
    elif hasattr(model, "gpt_neox"):
        return model.gpt_neox.layers
    else:
        raise ValueError(f"don't know how to get layer list for {type(model)}")




def main():

    model_name = "EleutherAI/pythia-14m"
    cv = ReadingContrastVector(model_name=model_name)
    dataset = Dataset(positive=["I am happy", "I am great", "All is okay"], negative=["I am sad", "I am bad", "It is shit"])
    cv.train(dataset)
    alpha = 0
    model = ControlModel(model_name=model_name, layer_ids=[1,2,3,4])
    #model.set_control(control_vector=cv.directions, alpha=alpha)
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


    # cv = PCAContrastVector(model_name="EleutherAI/pythia-14m")
    # dataset = Dataset(positive=["I am happy", "I am great", "All is okay"], negative=["I am sad", "I am bad", "It is shit"])
    # cv.train(dataset)

if __name__ == "__main__":
    main()
