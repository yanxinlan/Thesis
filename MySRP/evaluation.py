from .inferer import Inferer
import json

def evaluate(model_name, prompt_template, test_set):
    inferer = Inferer(model_name, prompt_template)

    # extract datapoints
    with open(test_set, 'r') as file:
        lines = file.readlines()

    # get responses
    responses = []
    for line in lines:
        obj = json.loads(line)
        response_single = inferer(choices=obj['choices'], question=obj['question'])
        responses.append(response_single)
    # print(responses)

    # get correct answers
    correct_answers = []
    for line in lines:
        obj = json.loads(line)
        correct_answers.append(obj['answer'])
    # print(correct_answers)

    # calculate accuracy
    if len(responses) != len(correct_answers):
        raise ValueError("The lengths of the output and correct answer lists must be the same.")

    correct_count = sum(output == correct for output, correct in zip(responses, correct_answers))
    accuracy = correct_count / len(responses) * 100

    return accuracy

prompt_template_medqa = {
    "description": "Default Template for medQA.",
    "primer": "Below is an instruction that describes a mulitple choice task, paired with choices. Give the correct answer that answers the question.",
    "question": "\n\n### question:\n",
    "choices": "\n\n### choices:\n",
    "answer": "\n\n### answer:\n",
    "answer_split": "### answer:"
}

accuracy_llama_medqa = evaluate(model_name="meta-llama/Llama-2-7b-chat-hf", prompt_template=prompt_template_medqa, test_set='data/medqa_test_clean.jsonl')
print(accuracy_llama_medqa)