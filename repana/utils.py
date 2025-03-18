from .controlModel import ControlModel
from .controlVector import ControlVector
from transformers import PreTrainedModel
import dataclasses
from typing import List, Dict, Literal
import torch
import numpy as np
from sklearn.decomposition import PCA
import torch
import polars as pl


@dataclasses.dataclass
class Dataset:
    positive: List[str]
    negative: List[str] | None = None

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
        max_prob_index = torch.argmax(probs[1])
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
        model_type: Literal["pythia", "mistral"],
        model: ControlModel,
        control_vector: ControlVector,
        alpha: float,
        normalize: bool,
        X: List = [],
        y: List = [],
        type: Literal["exact_match", "logit"] = "exact_match",
        settings: Dict = {},
        batch_size: int = 32,
        answer_list = []
    ):

    if type == "logit":
        
        print(f"Eval: {type}\nEvaluation function returning (results_df, accuracy)")

        settings["max_new_tokens"] = 1
        model.set_control(control_vector=control_vector.directions, alpha=alpha, normalize=normalize)
        model.tokenizer.padding_side = "right"
        answer_list_tokens = model.tokenizer(answer_list, return_tensors="pt", padding=True).input_ids.to(model.device)
        model.tokenizer.padding_side = "left"

        results = []
        
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]
            input_ids = model.tokenizer(batch_X, return_tensors="pt", padding=True).input_ids.to(model.device)

            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    return_dict_in_generate=True,
                    output_logits=True,
                    **settings)
            
            logits = output.logits[0]  # Shape: [batch_size, vocab_size]

            for j, question_logits in enumerate(logits):
                if model_type == 'mistral':
                    answer_logits = torch.stack([question_logits[token[1]] for token in answer_list_tokens])
                elif model_type == "pythia":
                    answer_logits = torch.stack([question_logits[token[0]] for token in answer_list_tokens])
                else:
                    print("Unknown model-type. Please use 'pythia' or 'mistral'")
                    break
                answer_probs = torch.softmax(answer_logits, dim=0).cpu().tolist()
                predicted_index = torch.argmax(answer_logits).item()
                predicted_answer = answer_list[predicted_index]
                
                result = {
                    "question": batch_X[j],
                    "correct_answer": batch_y[j],
                    "predicted_answer": predicted_answer,
                    "is_correct": batch_y[j] == predicted_answer,
                    "answer_logits": answer_logits,
                    "answer_probabilities": answer_probs
                }
                
                # Add individual probabilities for each answer
                for answer, prob in zip(answer_list, answer_probs):
                    result[f"prob_{answer}"] = float(prob)
                
                # Add logits for each answer
                # for answer, logit in zip(answer_list, answer_logits.cpu().tolist()):
                #     result[f"logit_{answer}"] = logit
                
                results.append(result)

        results_df = pl.DataFrame(results)
        accuracy = results_df["is_correct"].mean()

        return results_df, accuracy
   
    elif type == "exact_match":

        # Initialize lists to store results and accuracy
        results = []

        # Set the control vector
        model.set_control(control_vector=control_vector.directions, alpha=alpha, normalize=normalize)

        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]

            # Tokenize input
            input_ids = model.tokenizer(batch_X, return_tensors="pt", padding=True).input_ids.to(model.device)

            # Generate outputs
            with torch.no_grad():
                outputs = model.generate(input_ids, **settings)

            # Get only the newly generated tokens
            new_tokens = outputs[:, input_ids.shape[1]:]

            # Decode only the new generated tokens
            predicted_tokens = model.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

            # Compare predictions with ground truth
            for pred, true in zip(predicted_tokens, batch_y):
                pred = pred.strip().lower()
                true = true.strip().lower()
                results.append((true, pred))
        
        results_df = pl.DataFrame({
            "y": [r[0] for r in results],
            "prediction": [r[1] for r in results],
            "correct": [r[0] in r[1] for r in results]
        })

        # Calculate overall accuracy
        accuracy = results_df["correct"].sum() / len(results_df) if len(results_df) > 0 else 0
        print(results_df)
        print(f"Exact Match Accuracy: {accuracy:.4f}")
        
        return results_df, accuracy
    
    else:

        print("Invalid benchmark metric. Use either 'exact_match' or 'logit'.")
        
        return None, None


def eval_kld(
    model: ControlModel,
    control_vector: ControlVector,
    alpha: float,
    normalize: bool,
    X: List = [],
    y: List = [],
    settings: Dict = {},
    batch_size: int = 32,
    ):

    print(f"Computing D_kl \nEvaluation function returning (mean_kld, var_kld)")

    settings["max_new_tokens"] = 1
    kl_divergences = []

    for i in range(0, len(X), batch_size):
        batch_X = X[i:i+batch_size]
        batch_y = y[i:i+batch_size]
        
        input_ids = model.tokenizer(batch_X, return_tensors="pt", padding=True).input_ids.to(model.device)

        # reference distribution
        model.set_control(control_vector=control_vector.directions, alpha=0, normalize=normalize)
        with torch.no_grad():
            output = model.generate(
                input_ids,
                return_dict_in_generate=True,
                output_logits=True,
                **settings)
        
        logits = output.logits[-1]
        reference_probs_distribution = torch.nn.functional.softmax(logits, dim=-1)

        # alpha-modified distributions
        model.set_control(control_vector=control_vector.directions, alpha=alpha, normalize=normalize)
        with torch.no_grad():
            output = model.generate(
                input_ids,
                return_dict_in_generate=True,
                output_logits=True,
                **settings)
        
        logits = output.logits[-1]
        modified_probs_distribution = torch.nn.functional.softmax(logits, dim=-1)

        # Calculate KL divergence for each pair of distributions
        for i, (ref_dist, mod_dist) in enumerate(zip(reference_probs_distribution, modified_probs_distribution)):
            epsilon = 1e-8
            kl_div = torch.sum(ref_dist * (torch.log(ref_dist + epsilon) - torch.log(mod_dist + epsilon)))
            kl_divergences.append(kl_div.item())
    # Calculate and print the mean KL divergence
    mean_kl_divergence = float(sum(kl_divergences) / len(kl_divergences))
    var_kl_divergence = float(sum((kl - mean_kl_divergence) ** 2 for kl in kl_divergences) / len(kl_divergences))

    return mean_kl_divergence, var_kl_divergence


def eval_prob_mass(
    model_type: Literal['pythia', 'mistral'],
    model: ControlModel,
    control_vector: ControlVector,
    alpha: float,
    normalize: bool,
    X: List = [],
    y: List = [],
    settings: Dict = {},
    batch_size: int = 32,
    answer_list = [],
    mask_to_answer_list=False,
    ):

    print(f"Computing Probability mass \nEvaluation function returning (mean_prob_mass, var_prob_mass)")

    settings["max_new_tokens"] = 1

    corrects = []
    incorrects = []

    model.tokenizer.padding_side = "right"
    answer_list_tokens = model.tokenizer(answer_list, return_tensors="pt", padding=True).input_ids.to(model.device)
    model.tokenizer.padding_side = "left"

    if model_type == 'pythia':
        answer_list_tokens = answer_list_tokens[:, 0]
    elif model_type == 'mistral':
        answer_list_tokens = answer_list_tokens[:, 1]
    else:
        print("Unknown model type. Use either 'pythia' or 'mistral'.")

    for i in range(0, len(X), batch_size):
        batch_X = X[i:i+batch_size]
        batch_y = y[i:i+batch_size]
        input_ids = model.tokenizer(batch_X, return_tensors="pt", padding=True).input_ids.to(model.device)

        # alpha-modified distributions
        model.set_control(control_vector=control_vector.directions, alpha=alpha, normalize=normalize)
        with torch.no_grad():
            output = model.generate(
                input_ids,
                return_dict_in_generate=True,
                output_logits=True,
                **settings)
        
        logits = output.logits[-1]

        correct_answer_indices = [
            answer_list.index(y)
            for y in batch_y
        ]

        incorrect_answer_indices = [
            [i for i in range(len(answer_list)) if i != answer_list.index(y)]
            for y in batch_y
        ]

        if mask_to_answer_list:
            logits = logits[:, answer_list_tokens]
            modified_probs_distribution = torch.nn.functional.softmax(logits, dim=-1)

            correct_probs = [
                modified_probs_distribution[idx, correct_answer_indices[idx]].item()
                for idx in range(len(batch_y))
            ]

            incorrect_probs = [1 - c for c in correct_probs]

            corrects.extend(correct_probs)
            incorrects.extend(incorrect_probs)

        else:
            modified_probs_distribution = torch.nn.functional.softmax(logits, dim=-1)

            correct_probs = [
                modified_probs_distribution[idx, answer_list_tokens[correct_answer_indices[idx]]].item()
                for idx in range(len(batch_y))
            ]

            incorrect_probs = [
                sum(modified_probs_distribution[idx, answer_list_tokens[incorrect_answer_indices[idx]]].cpu().numpy())
                for idx in range(len(batch_y))
            ]

            corrects.extend(correct_probs)
            incorrects.extend(incorrect_probs)

    # Calculate and print the mean KL divergence
    mean_correct_prob = float(sum(corrects) / len(corrects))
    var_correct_prob = float(sum((co - mean_correct_prob) ** 2 for co in corrects) / len(corrects))

    mean_incorrect_prob = float(sum(incorrects) / len(incorrects))
    var_incorrect_prob = float(sum((co - mean_incorrect_prob) ** 2 for co in incorrects) / len(incorrects))

    return mean_correct_prob, var_correct_prob, mean_incorrect_prob, var_incorrect_prob


def eval_entropy(
    model: ControlModel,
    control_vector: ControlVector,
    alpha: float,
    normalize: bool,
    X: List = [],
    y: List = [],
    settings: Dict = {},
    batch_size: int = 32,
    ):

    print(f"Computing entropy \nEvaluation function returning (mean_kld, var_kld)")

    settings["max_new_tokens"] = 1
    entropies = []

    for i in range(0, len(X), batch_size):
        batch_X = X[i:i+batch_size]
        batch_y = y[i:i+batch_size]
        
        input_ids = model.tokenizer(batch_X, return_tensors="pt", padding=True).input_ids.to(model.device)

        # alpha-modified distributions
        model.set_control(control_vector=control_vector.directions, alpha=alpha, normalize=normalize)
        with torch.no_grad():
            output = model.generate(
                input_ids,
                return_dict_in_generate=True,
                output_logits=True,
                **settings)
        
        logits = output.logits[-1]
        modified_probs_distribution = torch.nn.functional.softmax(logits, dim=-1)

        # Calculate entropy for each distribution
        for i, mod_dist in enumerate(modified_probs_distribution):
            epsilon = 1e-8
            entropy = -torch.sum(mod_dist * torch.log2(mod_dist + epsilon))
            entropies.append(entropy.item())

    # Calculate and print the mean KL divergence
    mean_entropy = float(sum(entropies) / len(entropies))
    var_entropy = float(sum((h - mean_entropy) ** 2 for h in entropies) / len(entropies))

    return mean_entropy, var_entropy