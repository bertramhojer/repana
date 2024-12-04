from .controlModel import ControlModel
from .controlVector import ControlVector

import re
import torch
import numpy as np
import polars as pl

from dataclasses import dataclass
from tqdm import tqdm
from typing import Literal, List, Dict, Callable, Union, Optional, Tuple


# Helper function to save results
def save_results(df: pl.DataFrame, path: str, format: str = "csv"):
    """
    Save results in the specified format, handling nested data appropriately
    
    Args:
        df: DataFrame with results
        path: Path to save the file
        format: Output format ("csv" or "json")
    """
    if format == "csv":
        # Ensure all columns are CSV-safe
        df.write_csv(path)
    elif format == "json":
        # JSON can handle nested structures
        df.write_json(path)
    else:
        raise ValueError(f"Unsupported format: {format}")


@dataclass
class Dataset:

    def __init__(self, positive, negative=None):
        if negative is not None:
            assert len(positive) == len(negative), "Positive and negative datasets must have the same length"
        self.positive = positive
        self.negative = negative



@dataclass
class AnswerExtractor:
    """Defines how to extract and compare answers from model outputs"""
    extract_fn: Callable
    compare_fn: Callable
    preprocessing_fn: Optional[Callable] = None
    
    def extract(self, output: Union[str, torch.Tensor]) -> any:
        """Extract answer from model output"""
        return self.extract_fn(output)  # Fixed from extract_fc to extract_fn
    
    def compare(self, prediction: any, ground_truth: any) -> bool:
        """Compare prediction with ground truth"""
        if self.preprocessing_fn:
            prediction = self.preprocessing_fn(prediction)
            ground_truth = self.preprocessing_fn(ground_truth)
        return self.compare_fn(prediction, ground_truth)



def create_logit_extractor(tokenizer, answer_list, model_type: str = "mistral", device="cpu"):
    def extract_logit_answer(logits: torch.Tensor) -> str:
        answer_list_tokens = tokenizer(answer_list, return_tensors="pt", padding=True).input_ids.to(device)
        if model_type == 'mistral':
            answer_logits = torch.stack([logits[token[1]] for token in answer_list_tokens])
        elif model_type == "pythia":
            answer_logits = torch.stack([logits[token[0]] for token in answer_list_tokens])
        else:
            raise ValueError("Unknown model-type. Use 'pythia' or 'mistral'")
        
        predicted_index = torch.argmax(answer_logits).item()
        return answer_list[predicted_index], {
            'logits': answer_logits,
            'probabilities': torch.softmax(answer_logits, dim=0)
        }
    
    return AnswerExtractor(
        extract_fn=extract_logit_answer,
        compare_fn=lambda pred, true: pred[0] == true
    )




def create_exact_match_extractor():
    """Creates an extractor for exact match evaluation"""
    return AnswerExtractor(
        extract_fn=lambda x: x,
        compare_fn=lambda pred, true: true.strip().lower() in pred.strip().lower(),
        preprocessing_fn=lambda x: str(x).strip().lower()
    )


def create_gsm_extractor():
    """
    Creates an extractor for GSM-style numerical answers
    """
    def extract_numerical_answer(response: str) -> float:
        patterns = [
            r"#### (\-?[\d,\.]+)",  # Standard GSM8K format
            r"The answer is (\-?[\d,\.]+)",  # Alternative format
            r"Therefore,.*?(\-?[\d,\.]+)",  # Therefore format
            r"= (\-?[\d,\.]+)(?!\d)",  # Equation format
            r"(\-?[\d,\.]+)(?:\s*|$)"  # Bare number format
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response)
            if matches:
                try:
                    return float(matches[-1])
                except ValueError:
                    continue
        raise ValueError("No numerical answer found in response")
    
    return AnswerExtractor(
        extract_fn=extract_numerical_answer,
        compare_fn=lambda pred, true: abs(pred - float(true)) < 1e-6,
        preprocessing_fn=None
    )



def evaluate(
    model: "ControlModel",
    control_vector: "ControlVector",
    X: List,
    y: List,
    answer_extractor: AnswerExtractor,
    alpha: float,
    normalize: bool,
    settings: Dict = {},
    batch_size: int = 32,
    use_logits: bool = False
) -> Tuple[pl.DataFrame, float]:
    model.set_control(control_vector=control_vector.directions, alpha=alpha, normalize=normalize)
    results = []
    
    for i in range(0, len(X), batch_size):
        batch_X = X[i:i+batch_size]
        batch_y = y[i:i+batch_size]
        input_ids = model.tokenizer(batch_X, return_tensors="pt", padding=True).input_ids.to(model.device)
        
        with torch.no_grad():
            if use_logits:
                output = model.generate(
                    input_ids,
                    return_dict_in_generate=True,
                    output_logits=True,
                    **settings)
                model_outputs = output.logits[0]
            else:
                outputs = model.generate(input_ids, **settings)
                full_outputs = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                model_outputs = [
                    full_out[len(prompt):].strip() 
                    for prompt, full_out in zip(batch_X, full_outputs)
                ]
        
        for j, (output, true_answer) in enumerate(zip(model_outputs, batch_y)):
            try:
                prediction = answer_extractor.extract(output)
                is_correct = answer_extractor.compare(
                    prediction[0] if isinstance(prediction, tuple) else prediction,
                    true_answer
                )
                
                # Create base result with CSV-safe types
                result = {
                    "question": str(batch_X[j]),
                    "correct_answer": str(true_answer),
                    "predicted_answer": str(prediction[0] if isinstance(prediction, tuple) else prediction),
                    "is_correct": bool(is_correct),
                    "full_output": str(output)
                }
                
                # Handle additional metrics if present
                if isinstance(prediction, tuple) and len(prediction) > 1:
                    for key, value in prediction[1].items():
                        if isinstance(value, torch.Tensor):
                            # Convert tensor to string representation of list
                            result[f"{key}_list"] = str(value.cpu().tolist())
                        else:
                            # Convert other types to string
                            result[key] = str(value)
                            
                results.append(result)
                
            except Exception as e:
                print(f"Error processing output for batch item {j}: {str(e)}")
                print(f"Output: {output}")
                print(f"True answer: {true_answer}")
                results.append({
                    "question": str(batch_X[j]),
                    "correct_answer": str(true_answer),
                    "predicted_answer": "",
                    "is_correct": False,
                    "error": str(e),
                    "full_output": str(output)
                })
    
    results_df = pl.DataFrame(results)
    accuracy = results_df["is_correct"].mean()
    
    return results_df, accuracy