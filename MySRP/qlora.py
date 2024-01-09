from transformers import (AutoModelForCausalLM,
    BitsAndBytesConfig,
    LlamaTokenizer)
import torch
import os

os.environ["HUGGING_FACE_HUB_TOKEN"] = "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"

def load_hf_model(
    base_model,
    mode=8,
    gradient_checkpointing=False,
    device_map="auto",
):
    kwargs = {"device_map": device_map}
    if mode == 8:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=0.0,
        )
    elif mode == 4:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    elif mode == 16:
        kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)

    # setup tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference
    return model, tokenizer

model, tokenizer = load_hf_model(
    "meta-llama/Llama-2-7b-chat-hf",
    mode=4,
    gradient_checkpointing=False,
    device_map='auto')

# try inference
from transformers import GenerationConfig

sequences = ["<s>[INST] <<SYS>> You are a helpful assistant. <</SYS>>\
Extract the place names from the given sentence. [\INST]\n\
The capital of the United States is Washington D.C."]

inputs = tokenizer(sequences, padding=True, return_tensors="pt").to('cuda')

outputs = model.generate(
    **inputs,
    generation_config=GenerationConfig(
        do_sample=True,
        max_new_tokens=512,
        top_p=0.99,
        temperature=1e-8,
    )
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))