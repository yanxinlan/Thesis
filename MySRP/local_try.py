from datasets import load_dataset

df = load_dataset("bigbio/med_qa", "med_qa_en_4options_bigbio_qa", split="train")

print(df[0])