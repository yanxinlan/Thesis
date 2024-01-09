from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


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