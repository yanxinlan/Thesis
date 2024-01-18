from datasets import load_dataset
import jsonlines
import transformers

medqa_path = "bigbio/med_qa"
# model_name = "meta-llama/Llama-2-7b-chat-hf"
medqa_train = load_dataset(medqa_path, "med_qa_en_4options_bigbio_qa", split="train")
medqa_validation = load_dataset(medqa_path, "med_qa_en_4options_bigbio_qa", split="validation")
medqa_test = load_dataset(medqa_path, "med_qa_en_4options_bigbio_qa", split="test")

input_file = medqa_train
output_file = '/home/xyan/Thesis/data/medqa_train_clean.jsonl'
keys_to_keep = ['question', 'choices', 'answer']

with jsonlines.open(input_file, 'r') as reader, jsonlines.open(output_file, 'w') as writer:
    for obj in reader:
        new_dict = {}

        for key in keys_to_keep:
            if key in obj:
                if key == 'question':
                    # Keep the value unchanged
                    new_dict[key] = obj[key]
                elif isinstance(obj[key], list):
                    # Convert the list to a comma-separated string
                    new_dict[key] = ','.join(obj[key])
                else:
                    # Keep the value unchanged for non-list values
                    new_dict[key] = obj[key]

        writer.write(new_dict)

print(output_file)
print(output_file[0])