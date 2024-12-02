from . controlModel import ControlModel
from .controlVector import ControlVector

import re
import torch
import polars as pl

from dataclasses import dataclass
from tqdm import tqdm
from typing import Literal, List, Dict, Callable, Union, Optional, Tuple


@dataclass
class Dataset:

    def __init__(self, positive, negative=None):
        if negative is not None:
            assert len(positive) == len(negative), "Positive and negative datasets must have the same length"
        self.positive = positive
        self.negative = negative



@dataclass
class AnswerExtractor:
    """
    Defines how to extract and compare answers from model outputs
    """
    extract_fn: Callable
    compare_fn: Callable
    preprocessing_fn: Optional[Callable] = None

    def extract(self, output: Union[str, torch.Tensor]) -> any:
        return self.extract_fc(output)
    
    def compare(self, prediction: any, ground_truth: any) -> bool:
        if self.preprocessing_fn:
            prediction = self.preprocessing_fn(prediction)
            ground_truth = self.preprocessing_fn(ground_truth)
        return self.compare_fn(prediction, ground_truth)



def create_logit_extractor(tokenizer, answer_list, model_type: str = "mistral"):
    """
    Create extractor for logit-based evaluation
    """
    def extract_logit_answer(logits: torch.Tensor) -> str:
        answer_list_tokens = tokenizer(answer_list, return_tensors="pt", padding=True).input_ids.to(tokenizer.device)
        if model_type == 'mistral':
            answer_logits = torch.stack([logits[token[1]] for token in answer_list_tokens])
        elif model_type == "pythia":
            answer_logits = torch.stack([logits[token[0]] for token in answer_list_tokens])
        else:
            raise ValueError("Unknown model-type. Please use 'pythia' or 'mistral'")
        
        predicted_index = torch.argmax(answer_logits).item()
        return answer_list[predicted_index], {
            'logits': answer_logits,
            'probabilities': torch.softmax(answer_logits, dim=0)
        }
    
    return AnswerExtractor(
        extract_fn=extract_logit_answer,
        compare_fn=lambda pred, true: pred[0] == true,
        preprocessing_fn=None
    )



def create_exact_match_extractor():
    """
    Creates an extractor for exact match evaluation
    """
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
    """
    Args:
    model: The control model to evaluate
    control_vector: The control vector to apply
    X: List of input prompts
    y: List of ground truth answers
    answer_extractor: AnswerExtractor instance defining how to extract and compare answers
    alpha: Control strength
    normalize: Whether to normalize the control vector
    settings: Generation settings
    batch_size: Batch size for evaluation
    use_logits: Whether to use logits for evaluation (for token classification tasks)
    """