from dataclasses import dataclass
from typing import List, Optional, Union, Callable
from pathlib import Path
import polars as pl

@dataclass
class DataConfig:
    """Simple configuration for data loading"""
    input_columns: Union[str, List[str]]  # Column(s) to use as input
    target_column: str  # Column containing target/label
    template: Optional[str] = None  # Optional template for multiple input columns


@dataclass
class Dataset:

    def __init__(self, positive, negative=None):
        if negative is not None:
            assert len(positive) == len(negative), "Positive and negative datasets must have the same length"
        self.positive = positive
        self.negative = negative

    
class RepanaDataLoader:
    """Data loader with basic core functionality"""
    
    def __init__(
        self, 
        path: Union[str, Path],
        input_columns: Union[str, List[str]],
        target_column: str,
        template: Optional[str] = None,
        training: bool = False
    ):
        self.path = Path(path)
        self.config = DataConfig(
            input_columns=input_columns,
            target_column=target_column,
            template=template
        )
        self.training = training
    
    def load(self) -> List[str]:
        """Load and process data from CSV"""
        # Read the CSV file
        df = pl.read_csv(self.path)
        
        # Handle input columns
        if isinstance(self.config.input_columns, str):
            # Single input column
            inputs = df[self.config.input_columns].to_list()
        else:
            # Multiple input columns
            if self.config.template:
                # Use template if provided
                inputs = [
                    self.config.template.format(**row)
                    for row in df.select(self.config.input_columns).to_dicts()
                ]
            else:
                # Default to space-separated concatenation
                inputs = [
                    " ".join(str(x) for x in row)
                    for row in df.select(self.config.input_columns).rows()
                ]
        
        # Get targets
        targets = df[self.config.target_column].to_list()
        
        if self.training:
            # For training, combine inputs and targets into single strings
            return [f"{input} {target}" for input, target in zip(inputs, targets)]
        else:
            # For evaluation, return just the inputs
            return inputs, targets
