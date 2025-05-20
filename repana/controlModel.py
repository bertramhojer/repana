from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from typing import Optional, Callable, Union
from dataclasses import dataclass
import torch
import numpy as np
from abc import ABC, abstractmethod
from sklearn.decomposition import PCA



class ControlModel(torch.nn.Module):

    def __init__(self, model_name, layer_ids, dynamic=False):
        super().__init__()
        self.model_name = model_name
        self.layer_ids = layer_ids
        print("Starting loadin'")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto")
        print("Model loaded succesfully")
        self.layers = model_layer_list(self.model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = "left" # decoder-only always uses left-padding
        self.dynamic = dynamic
        self._create_control_model()
        print("Control model created")
    

    def _create_control_model(self):
        if self.dynamic:
            for layer_id in self.layer_ids:
                layer = self.layers[layer_id]
                if not isinstance(layer, DynamicControlBlock):
                    self.layers[layer_id] = DynamicControlBlock(layer)
                self.layers[layer_id].set_control(DynamicBlockParams.default())
                
        else:
            for layer_id in self.layer_ids:
                layer = self.layers[layer_id]
                if not isinstance(layer, ControlBlock):
                    self.layers[layer_id] = ControlBlock(layer)
                self.layers[layer_id].set_control(ControlBlockParams.default())
    

    def set_control(self, control_vector, alpha: float = 1.0, kappa: float = 1.0, normalize: bool = False, operator: Callable = torch.add):
        directions = control_vector.directions if hasattr(control_vector, 'directions') else control_vector

        if self.dynamic:
            for layer_id in self.layer_ids:
                params = DynamicBlockParams(
                    control = directions[layer_id],
                    kappa = kappa,
                    normalize = normalize,
                    operator = operator
                )
                self.layers[layer_id].set_control(params)
        else:
            for layer_id in self.layer_ids:
                params = ControlBlockParams(
                    control=directions[layer_id],
                    alpha=alpha,
                    normalize=normalize,
                    operator=operator
                )
                self.layers[layer_id].set_control(params)
    
    def reset_control(self):
        for layer_id in self.layer_ids:
            if isinstance(self.layers[layer_id], ControlBlock):
                self.layers[layer_id].reset()


    @property
    def device(self) -> torch.device:
        return self.model.device

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)



@dataclass
class ControlBlockParams:
    control: Optional[Union[torch.Tensor, np.ndarray]] = None
    alpha: float = 0.0
    normalize: bool = False
    operator: Callable = torch.add

    @classmethod
    def default(cls):
        return cls()

@dataclass
class DynamicBlockParams:
    control: Optional[Union[torch.Tensor, np.ndarray]] = None
    kappa: float = 1.0
    normalize: bool = False
    operator: Callable = torch.add

    @classmethod
    def default(cls):
        return cls()


class ControlBlock(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.block = layer
        self.params: ControlBlockParams = ControlBlockParams.default()

    def set_control(self, params: ControlBlockParams) -> None:
        self.params = params

    def reset(self) -> None:
        self.set_control(ControlBlockParams.default())

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        
        if self.params.control is None:
            return output
        
        if isinstance(output, tuple):
            modified = output[0]
        else:
            modified = output

        control = self.params.control
        
        # Convert control to a PyTorch tensor if it's a NumPy array
        if isinstance(control, np.ndarray):
            control = torch.from_numpy(control).float()

        if len(control.shape) == 1:
            control = control.reshape(1, 1, -1)

        assert len(control.shape) == len(modified.shape), f"Control vector shape {control.shape} doesn't match output shape {modified.shape}"
        
        # Ensure control is on the same device as modified
        control = control.to(modified.device)
        
        norm_pre = torch.norm(modified, dim=-1, keepdim=True)
        
        modified = self.params.operator(modified, control * self.params.alpha)
        
        if self.params.normalize:
            norm_post = torch.norm(modified, dim=-1, keepdim=True)
            modified = modified / norm_post * norm_pre

        if isinstance(output, tuple):
            output = (modified,) + output[1:]
        else:
            output = modified

        return output


class DynamicControlBlock(torch.nn.Module):
    def __init__(self, block, kappa: float = 1.0):
        super().__init__()
        self.block = block
        self.params: DynamicBlockParams = DynamicBlockParams.default()

    def set_control(self, params: DynamicBlockParams) -> None:
        self.params = params

    def forward(self, hidden_states, *args, **kwargs):
        # 1) Run the underlying block
        base_output = self.block(hidden_states, *args, **kwargs)

        # 2) Unpack tensor vs. tuple
        if isinstance(base_output, tuple):
            core_out, *rest = base_output
        else:
            core_out = base_output
            rest = []

        # 3) Prepare control vector
        cv = self.params.control
        if isinstance(cv, np.ndarray):
            cv = torch.from_numpy(cv).float()
        # shape (1, 1, hidden_dim)
        if cv.dim() == 1:
            cv = cv.view(1, 1, -1)
        cv = cv.to(hidden_states.device)
        # broadcast to (batch, seq_len, hidden_dim)
        cv = cv.expand_as(hidden_states)

        # 4) Compute gate λ
        delta = cv - hidden_states
        delta_norm = delta.norm(dim=-1, keepdim=True)  # (B, T, 1)
        lambda_r = delta_norm / (delta_norm + self.params.kappa)
        lambda_r = lambda_r.clamp(0.0, 1.0)

        print(delta_norm[:,0,0], lambda_r[:,0,0])

        # 5) Interpolate **the tensor** core_out ↔ cv
        controlled = (1 - lambda_r) * core_out + lambda_r * cv

        # 6) Re-package outputs exactly as original block did
        if rest:
            return (controlled, *rest)
        else:
            return controlled

    


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