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
medqa_train = load_dataset(medqa_path, "med_qa_en_4options_bigbio_qa", split="train")
print(medqa_train[0])
medqa_test = load_dataset(medqa_path, "med_qa_en_4options_bigbio_qa", split="test")
print(medqa_test[0])

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
base_model = AutoModelForCausalLM.from_pretrained(model_name)

# TODO: train and test dataset

device = torch.device("cuda")
base_model.to(device)

# inference function

def inference(text, model, tokenizer, max_input_tokens=1000, max_output_tokens=100):
    # tokenize
    input_ids = tokenizer.encode(
        text,
        return_tensors="pt",
        max_length=max_input_tokens
    )

    # generate
    device = model.device
    generated_tokens_with_prompt = model.generate(
        input_ids=input_ids.to(device),
        max_length=max_output_tokens
    )

    # decode
    generated_text_with_prompt = tokenizer.batch_decode(generated_tokens_with_prompt, skip_special_tokens=True)

    # strip the prompt
    generated_text_answer = generated_tokens_with_prompt[0][len(text):]

    return generated_text_answer

# test with base model

test_text = medqa_test[0][['question', 'type', 'choices']]
print('Question input (test):', test_text)
print(f'Correct answer (test): {medqa_test[0]["answer"]}')
print("model's answer: ")
print(inference(test_text, base_model, tokenizer))