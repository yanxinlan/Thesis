import sys
sys.path.append('/home/xyan/Thesis/MySRP/')
from evaluation import evaluate
from datetime import datetime

prompt_template_medqa = {
    "description": "Default Template for medQA.",
    "primer": "Below is an instruction that describes a multiple choice task, paired with choices. Pick the correct answer without modification or explanations.",
    "question": "\n\n### question:\n",
    "choices": "\n\n### choices:\n",
    "answer": "\n\n### answer:\n",
    "answer_split": "### answer:"
}

print(datetime.now())
accuracy_llama_medqa = evaluate(
    model_name="meta-llama/Llama-2-7b-chat-hf",
    prompt_template=prompt_template_medqa,
    test_set='data/medqa_test_clean.jsonl',
    responses_path="results/llama_medqa_responses.json",
    answers_path="results/llama_medqa_correct_answers.json"
)
print(datetime.now())