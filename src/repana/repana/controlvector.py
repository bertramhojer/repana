from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from .controlModel import ControlModel
import dataclasses
from typing import List
import torch
import json
import sys
import os
import numpy as np
import tqdm
from abc import ABC, abstractmethod
from sklearn.decomposition import PCA
import importlib


@dataclasses.dataclass
class ControlVector(ABC):
    model_name: str
    directions: dict[int, np.ndarray] = dataclasses.field(default_factory=dict)


    @abstractmethod
    def train(self, dataset, vector):
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
    

    def save(self, path):
        save_data = {
            "model_name": self.model_name,
            "directions": {str(k): {
                "data": v.tolist(),
                "dtype": str(v.dtype)
            } for k, v in self.directions.items()},
            "class_name": self.__class__.__name__,
            "module_name": self.__class__.__module__
        }
        
        with open(path, 'w') as f:
            json.dump(save_data, f)

        
    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            load_data = json.load(f)
        
        module_name = load_data['module_name']
        class_name = load_data['class_name']
        
        module = importlib.import_module(module_name)
        class_ = getattr(module, class_name)
        
        instance = class_(load_data['model_name'])
        
        # Load the directions with correct dtype
        instance.directions = {
            int(k): np.array(v['data'], dtype=np.dtype(v['dtype']))
            for k, v in load_data['directions'].items()
        }
        
        return instance
        



@dataclasses.dataclass
class ReadingVector(ControlVector):
    additional_param: float = 1.0

    def train(self, dataset):

        positive_representations, _ = self._read_representations(dataset)

        for layer in tqdm.tqdm(range(self.n_layers)):

            h = positive_representations[layer]
            self.directions[layer] = np.mean(h, axis=0)
        
        #print("##### ", self.directions)
        
        return self


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

        return self


    def additional_method(self):
        # Add any additional methods specific to the derived class
        pass
        

            
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