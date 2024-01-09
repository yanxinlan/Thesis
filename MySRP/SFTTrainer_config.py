from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments
import torch

## load data
df_medqa_path='/home/xyan/Thesis/data/MedQA/data_clean/questions/US/train.jsonl'
df_medqa=[]
with open(df_medqa_path, 'r') as file:
    for line in file:
        # Load each line as a JSON object
        json_data = json.loads(line)
        df_medqa.append(json_data)



## load model

## configuration
bnb_config = BitsAndBytesConfig(
  load_in_4bit = True,
  bnb_4bit_quant_type='nf4',
  vnv_4bit_compute_dtype=torch.float16
)

## download model

model_name = '/home/xyan/model/llama/llama2-7b-chat'
model = AutoModelForCausalLM.from_pretrained(
  model_name,
  quantization_config=bnb_config,
  trust_remote_code=True
)

model.config.use_cache = False


## download tokenizer

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

## peft config

lora_alpha=16
lora_dropout = 0.1
lora_r=64

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias='none',
    task_type='CAUSAL_LM'
)

## fine-tuning config

output_dir = './results'
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
optim = 'paged_adamw_32bit'
save_steps = 100
logging_steps = 10
learning_rate =2e-4
max_grad_norm = 0.3
max_steps = 100
warmup_ratio = 0.03
lr_scheduler_type = 'constant'

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    fp16=True,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type
)

# SFTTrainer

max_seq_length = 512

trainer = SFTTrainer(
    model=model,
    train_dataset=df_medqa,
    peft_config=peft_config,
    dataset_text_field='text',
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments
)

for name, module in trainer.model.named_modules():
    if 'norm' in name:
        module = module.to(torch.float32)

trainer.train()