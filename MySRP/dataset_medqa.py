from datasets import load_dataset
import jsonlines

medqa_path = "bigbio/med_qa"
# model_name = "meta-llama/Llama-2-7b-chat-hf"
medqa_train = load_dataset(medqa_path, "med_qa_en_4options_bigbio_qa", split="train")
medqa_validation = load_dataset(medqa_path, "med_qa_en_4options_bigbio_qa", split="validation")
medqa_test = load_dataset(medqa_path, "med_qa_en_4options_bigbio_qa", split="test")

input_files = [medqa_train, medqa_validation, medqa_test]
output_files = ['/home/xyan/Thesis/data/medqa_train_clean.json', '/home/xyan/Thesis/data/medqa_validation_clean.json', '/home/xyan/Thesis/data/medqa_test_clean.json']
keys_to_keep = ['question', 'choices', 'answer']

key_transformations = {
    'question': lambda x: x,
    'choices': lambda x: ','.join(x) if isinstance(x, list) else x,
    'answer': lambda x: ','.join(x) if isinstance(x, list) else x,
}

for input_file, output_file in zip(input_files, output_files):
    with jsonlines.open(output_file, 'w') as writer:
        for example in input_file:
            new_example = {key: key_transformations[key](example[key]) for key in key_transformations}
            writer.write(new_example)
