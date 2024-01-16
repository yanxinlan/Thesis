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

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
#from llama import BasicModelRunner

logger = logging.getLogger(__name__)
global_config = None

medqa_path = "bigbio/med_qa"
use_hf = True
model_name = "meta-llama/Llama-2-7b-chat-hf"
medqa_train = load_dataset(medqa_path, "med_qa_en_4options_bigbio_qa", split="train")
medqa_validation = load_dataset(medqa_path, "med_qa_en_4options_bigbio_qa", split="validation")
medqa_test = load_dataset(medqa_path, "med_qa_en_4options_bigbio_qa", split="test")

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

device = torch.device("cuda")
base_model.to(device)

# inference function

def inference(text, model, tokenizer, max_input_tokens=500, max_output_tokens=100):
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
    generated_text_answer = generated_text_with_prompt[0]

    return generated_text_answer

# test with base model

# question_keys = ['question', 'type', 'choices']
# test_text = {key: medqa_test[0][key] for key in question_keys}
# formatted_test_text = "Question: {question}\nType: {type}\nThe choices are: {choices}".format(**test_text)
#
# print('Question input (test):', formatted_test_text)
# print(f'Correct answer (test): {medqa_test[0]["answer"]}')
# print("model's answer: ")
# print(inference(formatted_test_text, base_model, tokenizer))

# training arguments
max_steps = 20
trained_model_name = f"medqa_{max_steps}_steps"
output_dir = trained_model_name

training_args = TrainingArguments(
    learning_rate=1e-5,
    num_train_epochs=1,
    max_steps=max_steps,
    per_device_train_batch_size=128,
    output_dir=output_dir,

    overwrite_output_dir=True,
    disable_tqdm=False,
    eval_steps=10,
    save_steps=10,
    warmup_steps=1,
    per_device_eval_batch_size=128,
    evaluation_strategy="steps",
    logging_strategy="steps",
    logging_steps=10,
    optim='adafactor',
    gradient_accumulation_steps=4,
    gradient_checkpointing=False,

    # early stopping
    load_best_model_at_end=True,
    save_total_limit=1,
    metric_for_best_model='eval_loss',
    greater_is_better=False
)

# trainer
trainer = Trainer(
    model=base_model,
    args=training_args,
    #total_steps=max_steps,
    train_dataset=medqa_train,
    eval_dataset=medqa_validation
)

# train
training_output = trainer.train()

# save
save_dir = f"{output_dir}/checkpoints"
trainer.save_model(save_dir)
print("Saved model to:", save_dir)

# load model
# finetuned_model = AutoModelForCausalLM.from_pretrained(save_dir, local_files_only=True)
# finetuned_model.to(device)

# test again
# question_keys = ['question', 'type', 'choices']
# test_text = {key: medqa_test[0][key] for key in question_keys}
# formatted_test_text = "Question: {question}\nType: {type}\nThe choices are: {choices}".format(**test_text)
#
# print('Question input (test):', formatted_test_text)
# print(f'Correct answer (test): {medqa_test[0]["answer"]}')
# print("model's answer: ")
# print(inference(formatted_test_text, finetuned_model, tokenizer))

