from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

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
    

    def evaluate(self, cv=None, alpha=None, X=None, y=None, sample=False, max_new_tokens=1, repetition_penalty_penalty=1.1):

        if alpha != None:
            self.set_control(control_vector=cv.directions, alpha=alpha)

        settings = {
            "pad_token_id": self.tokenizer.eos_token_id, # silence warning
            "do_sample": sample, # temperature=0
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": repetition_penalty_penalty, # reduce control jank
        }

        results = []
        for X, y in zip(X, y):
            input_ids = self.tokenizer(X, return_tensors="pt").input_ids.to(self.device)
            with torch.no_grad():
                outputs = self.generate(input_ids, **settings).squeeze()
            generated_token = outputs[-1]
            predicted_token = self.tokenizer.decode(generated_token).strip()
            expected_token = y
            results.append((expected_token, predicted_token))
        
        correct_predictions = sum(1 for expected, predicted in results if expected == predicted)
        total_predictions = len(results)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

        return results, accuracy


    

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