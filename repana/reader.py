from abc import ABC
import dataclasses
from typing import Dict, TYPE_CHECKING
import numpy as np
from repana import ControlModel
import os
import pickle
import numpy as np


@dataclasses.dataclass
class Reader(ABC):
    model_name: str
    device: str = 'cuda'
    directions: Dict[int, np.ndarray] = dataclasses.field(default_factory=dict)

    def _read_representations(self, prompt: str = "Hello, ", max_new_tokens: int = 20):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        self.prompt = prompt

        def model_layer_list(model: ControlModel | AutoModelForCausalLM) -> torch.nn.ModuleList:
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

        model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.pad_token_id = 0

        model_layers = model_layer_list(model)
        self.n_layers = len(model_layers)

        representations: dict = {}

        with torch.no_grad():

            if "qwen" in self.model_name.lower():

                messages = [
                    {"role": "user", "content": self.prompt}
                ]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
                )
            
                tokens = tokenizer([text], return_tensors="pt").to(model.device)
            else:
                tokens = tokenizer(self.prompt, return_tensors="pt").to(model.device)

            # Generate output tokens
            generated = model.generate(
                input_ids=tokens.input_ids,
                attention_mask=tokens.attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id
            )
            decoded_tokens = tokenizer.batch_decode(generated, skip_special_tokens=False)
            # Get hidden states for all tokens (input + generated)
            out = model(
                input_ids=generated,
                output_hidden_states=True
            )
            hidden_states = out.hidden_states[-self.n_layers:]
            for l in range(self.n_layers):
                token_representations: dict[int, np.ndarray] = {}
                for j in range(len(generated[0])):
                    token_representations[j] = hidden_states[l][0][j].cpu().numpy()
                representations[l] = token_representations

        self.tokens = tokenizer.convert_ids_to_tokens(generated[0])
        self.token_ids = generated
        self.decoded_tokens = decoded_tokens

        return representations


    def save(self, representations, example):
        save_data = {
            "model_name": self.model_name,
            "prompt": self.prompt,
            "tokens": self.tokens,
            "token_ids": self.token_ids,
            "output": self.decoded_tokens,
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
        

def get_token_idx(
    reader, layer: int, token: str
    ):
    lst = []
    for idx, t in enumerate(reader["tokens"]):
        if t == token:
            lst.append(idx)
    return lst

def get_token_representations(
    representations, layer: int, token_index: int
    ):
    return representations[layer][token_index]

def cosine_sim(A, B):
    cosine = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))
    return cosine


if __name__ == "__main__":
    reader = Reader(model_name="Qwen/Qwen3-0.6B", device="mps")
    representations = reader._read_representations("Do language model representations converge during reasoning trace generation?", max_new_tokens=1024)
    reader.save(representations, example="2plus2")
    loaded_data = Reader.load(example="2plus2")
    print(loaded_data["output"])
    #get_token_idx(reader, layer=5, token="world")
    print("all done!")
