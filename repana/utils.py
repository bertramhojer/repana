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
import torch
import polars as pl


@dataclasses.dataclass
class Dataset:

    def __init__(self, positive, negative=None):
        if negative is not None:
            assert len(positive) == len(negative), "Positive and negative datasets must have the same length"
        self.positive = positive
        self.negative = negative



def get_word_probability(tokens, probs, combination_method='product'):
    # Filter out padding tokens
    actual_tokens = tokens[tokens != 0]
    
    # Get probabilities for each token
    token_probs = [probs[0][token].item() for token in actual_tokens]
    
    # Combine probabilities based on the specified method
    if len(token_probs) == 1:
        return token_probs[0]
    elif combination_method == 'product':
        return torch.prod(torch.tensor(token_probs)).item()
    elif combination_method == 'geometric_mean':
        return torch.prod(torch.tensor(token_probs)).pow(1/len(token_probs)).item()
    else:
        raise ValueError("Invalid combination method. Choose 'product' or 'geometric_mean'.")


def assess_accuracy(probabilities, X, y, answer_list):
    results = []
    
    for question, correct_answer, probs in zip(X, y, probabilities):
        max_prob_index = np.argmax(probs)
        predicted_answer = answer_list[max_prob_index]
        
        is_correct = predicted_answer == correct_answer
        
        result = {
            "question": question,
            "correct_answer": correct_answer,
            "predicted_answer": predicted_answer,
            "is_correct": is_correct,
        }
        
        # Add probabilities for each answer
        for answer, prob in zip(answer_list, probs):
            result[f"prob_{answer}"] = prob
        
        results.append(result)
    
    # Create Polars DataFrame
    df = pl.DataFrame(results)
    
    return df




def evaluate(
    model: ControlModel,
    control_vector: ControlVector,
    alpha: float,
    normalize: bool,
    X: List = [],
    y: List = [],
    type: Literal["em", "logits"] = "em",
    task: Literal["ioi", "deduction"] = "ioi",
    settings: Dict = {},
    batch_size: int = 32,
    answer_list = []
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
            
            generated_tokens = outputs
            predicted_tokens = model.tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
            
            for predicted_token, expected_token in zip(predicted_tokens, batch_y):
                results.append((expected_token.strip().lower(), predicted_token.strip().lower()))
        
        correct_predictions = sum(1 for expected, predicted in results if expected in predicted)
        total_predictions = len(results)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

        return results, accuracy

    elif type == "logits":

        settings["max_new_tokens"] = 1
        #model.set_control(control_vector=control_vector.directions, alpha=alpha, normalize=normalize)
        results = []

        answer_list_tokens = model.tokenizer(answer_list, return_tensors="pt", padding=True).input_ids.to(model.device)
        results_df = pl.DataFrame()

        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]

            print(len(batch_X))
        
            input_ids = model.tokenizer(batch_X, return_tensors="pt", padding=True).input_ids.to(model.device)
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    return_dict_in_generate=True,
                    output_logits=True,
                    **settings)
            
            logits = output.logits[-1]
            probs = torch.nn.functional.softmax(logits, dim=-1)

            batch_results = []
            for question_probs in probs:
                answer_probs = [
                    get_word_probability(tokens, question_probs.unsqueeze(0), 'product')
                    for tokens in answer_list_tokens
                ]
                batch_results.append(answer_probs)

            results.extend(batch_results)

            batch_df = assess_accuracy(results, X, y, answer_list)
            results_df = results_df.vstack(batch_df)
        
        accuracy = results_df["is_correct"].sum() / len(results_df)
        
        return results_df, accuracy