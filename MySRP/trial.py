from datasets import load_dataset
import tempfile
import logging
import random
import config
import os
import yaml
import time
import torch
import transformers

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from llama import BasicModelRunner

logger = logging.getLogger(__name__)
global_config = None

medqa_path = "bigbio/med_qa"
use_hf = True
model_name = "meta-llama/Llama-2-7b-chat-hf"
medqa_dataset = load_dataset(medqa_path)
print(medqa_dataset[0])

training_config = {
    "model": {
        "pretrained_name": model_name,
        'max_length': 2048
    },
    'datasets': {
        'use_hf': use_hf,
        'path': medqa_path
    },
    'verbose': True
}

# load model

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)