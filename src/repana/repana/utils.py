from .controlModel import ControlModel
from .controlVector import ControlVector
from transformers import PreTrainedModel
import dataclasses
from typing import List, Dict, Literal
import torch
import json
import pickle
import os
import numpy as np
import tqdm
from sklearn.decomposition import PCA


@dataclasses.dataclass
class Dataset:

    def __init__(self, positive, negative=None):
        if negative is not None:
            assert len(positive) == len(negative), "Positive and negative datasets must have the same length"
        self.positive = positive
        self.negative = negative



def evaluate(
    model: ControlModel,
    control_vector: ControlVector,
    alpha: float,
    normalize: bool,
    X: List = [],
    y: List = [],
    type: Literal["em", "logits"] = "em",
    settings: Dict = {},
    batch_size: int = 32
    ):

    if type == "em":

        model.set_control(control_vector=control_vector.directions, alpha=alpha, normalize=normalize)

        results = []
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]
            
            input_ids = model.tokenizer(batch_X, return_tensors="pt", padding=True).input_ids.to(model.device)
            with torch.no_grad():
                outputs = model.generate(input_ids, **settings)
            
            generated_tokens = outputs[:, -1]
            predicted_tokens = model.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            
            for predicted_token, expected_token in zip(predicted_tokens, batch_y):
                results.append((expected_token, predicted_token.strip()))
        
        correct_predictions = sum(1 for expected, predicted in results if expected == predicted)
        total_predictions = len(results)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

        print("ACCURACY: ", accuracy)

        return results, accuracy

    elif type == "logits":

        ## IMPLEMENT LOGIT DIFFERENCE EVALUATION
        pass
