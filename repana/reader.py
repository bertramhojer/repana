from abc import ABC
import dataclasses
from typing import Dict, TYPE_CHECKING
import numpy as np
from repana import ControlModel
import os
import pickle

@dataclasses.dataclass
class Reader(ABC):
    model_name: str
    device: str = 'cuda'
    directions: Dict[int, np.ndarray] = dataclasses.field(default_factory=dict)
    revision: str | None = None  # Field for revision (optional for pythia models) e.g. "step10000"

    def _read_representations(self, prompt):
        """
        Read representations from the model for the given dataset.
        Checks whether there are positive and negative examples in the dataset.
        If there are, it reads the representations for the positive and negative examples and returns them.
        If there are no negative examples, it returns the representations for the positive examples and negative as {}.
        """

        from transformers import PreTrainedModel, AutoTokenizer, AutoModelForCausalLM
        import torch

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

        if self.revision is not None:
            model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto", revision=self.revision)
        else:
            model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.pad_token_id = 0

        model_layers = model_layer_list(model)
        self.n_layers = len(model_layers)

        representations: dict = {}

        with torch.no_grad():
            tokens = tokenizer(prompt, return_tensors="pt").to(model.device)
            out = model(**tokens, output_hidden_states=True)
            hidden_states = out.hidden_states[-self.n_layers:]
            for l in range(self.n_layers):
                token_representations: dict[int, np.ndarray] = {}
                for j in range(len(tokens.input_ids[0])):
                    token_representations[j] = hidden_states[l][0][j].cpu().numpy()
                representations[l] = token_representations
            
        return representations
    

    def save(self, representations, example):
        save_data = {
            "model_name": self.model_name,
            "representations": representations
            }
        
        path = os.path.join("data", "representations", f"{example}.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
    
    
    @classmethod
    def load(cls, example):
        path = os.path.join("data", "representations", f"{example}.pkl")
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data
                    
        
if __name__ == "__main__":
    reader = Reader(model_name="EleutherAI/pythia-14m")
    representations = reader._read_representations("Hello, world!")
    print("all done!")
