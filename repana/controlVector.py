from transformers import PreTrainedModel, AutoTokenizer, AutoModelForCausalLM
#from ctransformers import AutoModelForCausalLM
from .controlModel import ControlModel
import dataclasses
from typing import List, Dict, Optional
import torch
import json
import pickle
import sys
import os
import numpy as np
import tqdm
from abc import ABC, abstractmethod
from sklearn.decomposition import PCA
import importlib
from torch.utils.data import DataLoader



@dataclasses.dataclass
class ControlVector(ABC):
    model_name: str | List[str]
    standardize: bool
    device: str = 'cuda'
    batch_size: int = 32  # New field for batch size
    directions: Dict[int, np.ndarray] = dataclasses.field(default_factory=dict)
    base_dir: str = 'cv'  # Field for base directory

    def __post_init__(self):
        if isinstance(self.model_name, str) and "llama" in self.model_name:
            self.model_name, self.model_file = self.model_name.split(":")
        else:
            self.model_file = None

    @abstractmethod
    def train(self, dataset, vector):
        pass
    
    def _read_representations(self, dataset):
        """
        Read representations from the model for the given dataset.
        Checks whether there are positive and negative examples in the dataset.
        If there are, it reads the representations for the positive and negative examples and returns them.
        If there are no negative examples, it returns the representations for the positive examples and negative as {}.
        """

        model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Move model to the appropriate device (GPU if available)
        model = model.to(self.device)
        tokenizer.pad_token_id = 0

        # Get the number of layers in the model
        model_layers = model_layer_list(model)
        self.n_layers = len(model_layers)

        def process_batch(batch):
            # Process a single batch of data
            with torch.no_grad():
                inputs = tokenizer(batch, padding=True, return_tensors="pt").to(self.device)
                outputs = model(**inputs, output_hidden_states=True)
                # Get the hidden states from the last n_layers
                hidden_states = outputs.hidden_states[-self.n_layers:]
                # Return only the last token's representation for each layer
                return [layer[:, -1, :] for layer in hidden_states]

        def get_representations(data):
            # Initialize dictionary to store representations for each layer
            representations = {layer: [] for layer in range(self.n_layers)}
            # Create a DataLoader for efficient batching
            dataloader = DataLoader(data, batch_size=self.batch_size, shuffle=False)
            
            for batch in tqdm.tqdm(dataloader):
                batch_representations = process_batch(batch)
                for i in range(self.n_layers):
                    representations[i].append(batch_representations[i])
            
            # Concatenate all batches and convert to numpy arrays
            return {l: torch.cat(representations[l]).cpu().numpy() for l in range(self.n_layers)}

        # Process positive examples
        positive_representations = get_representations(dataset.positive)

        # Process negative examples if they exist
        if dataset.negative is not None:
            negative_representations = get_representations(dataset.negative)
        else:
            negative_representations = None

        return positive_representations, negative_representations
    

    def _compute_norm(self, positive_representations, negative_representations=None):

        if negative_representations is not None:
            representations = positive_representations - negative_representations
        else:
            representations = positive_representations

        mu_representations = np.mean(representations, axis=0)
        # Compute the norm of the mu_representations vector
        norm = np.linalg.norm(mu_representations)
        
        return norm
        

    def save(self, task: str, cv_type: str, shots: int):
        save_data = {
            "model_name": self.model_name,
            "standardize": self.standardize,
            "directions": {str(k): {
                "data": v.tolist(),
                "dtype": str(v.dtype)
            } for k, v in self.directions.items()},
            "class_name": self.__class__.__name__,
            "module_name": self.__class__.__module__
        }
        
        # Construct the full path
        model_name = self.model_name.split('/')[-1] if isinstance(self.model_name, str) else self.model_name[-1].split('/')[-1]
        filename = f"{model_name}-{shots}.pkl"
        full_path = os.path.join(self.base_dir, task, cv_type, filename)
        
        # Create subdirectories if they don't exist
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        with open(full_path, 'wb') as f:
            pickle.dump(save_data, f)

    @classmethod
    def load(cls, path: str):
        with open(path, 'rb') as f:
            load_data = pickle.load(f)
        
        module_name = load_data['module_name']
        class_name = load_data['class_name']
        
        module = importlib.import_module(module_name)
        class_ = getattr(module, class_name)
        
        instance = class_(load_data['model_name'], load_data['standardize'])
        
        # Load the directions with correct dtype
        instance.directions = {
            int(k): np.array(v['data'], dtype=np.dtype(v['dtype']))
            for k, v in load_data['directions'].items()}

        return instance


@dataclasses.dataclass
class ReadingVector(ControlVector):
    additional_param: float = 1.0

    def train(self, dataset):

        positive_representations, _ = self._read_representations(dataset)

        for layer in tqdm.tqdm(range(self.n_layers)):

            h = positive_representations[layer]
            self.directions[layer] = np.mean(h, axis=0)
        
        return self



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
class PCAReadingVector(ControlVector):
    additional_param: float = 1.0

    def train(self, dataset):

        positive_representations, _ = self._read_representations(dataset)

        for layer in tqdm.tqdm(range(self.n_layers)):

            h = positive_representations[layer]

            pca_model = PCA(n_components=1).fit(h)

            control_vector = pca_model.components_.astype(np.float32).squeeze(axis=0)

            if self.standardize:
                norm = self._compute_norm(h)
                control_vector *= norm
            
            self.directions[layer] = control_vector

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
            control_vector = pca_model.components_.astype(np.float32).squeeze(axis=0)

            if self.standardize:
                norm = self._compute_norm(h)
                control_vector *= norm
            
            self.directions[layer] = control_vector

        return self
        

            
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