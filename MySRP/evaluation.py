import sys
sys.path.append('/home/xyan/Thesis/MySRP/')
from inferer import Inferer
import json

def evaluate(model_name, prompt_template, test_set, responses_path, answers_path):
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
    with open(responses_path, 'w') as file:
        json.dump(responses, file)

    # get correct answers
    correct_answers = []
    for line in lines:
        obj = json.loads(line)
        correct_answers.append(obj['answer'])
    # print(correct_answers)
    with open(answers_path, 'w') as file:
        json.dump(correct_answers, file)

    # calculate accuracy
    if len(responses) != len(correct_answers):
        raise ValueError("The lengths of the output and correct answer lists must be the same.")

    correct_count = sum(correct in output for output, correct in zip(responses, correct_answers))
    accuracy = correct_count / len(responses) * 100

    print(f"The accuracy is {accuracy}%")

    return accuracy

