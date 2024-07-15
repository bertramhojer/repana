from transformers import AutoModelForCausalLM, AutoTokenizer
from myrep import ControlVector, ControlModel, DatasetEntry
import dataclasses
from typing import List
import torch
import json
import pickle
import os
import numpy as np
import tqdm