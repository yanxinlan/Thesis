from .inferer import Inferer

prompt_template = {
    "description": "Default Template for medQA.",
    "primer": "Below is an instruction that describes a mulitple choice task, paired with choices. Give the correct answer that answers the question.",
    "question": "\n\n### question:\n",
    "choices": "\n\n### choices:\n",
    "answer": "\n\n### answer:\n",
    "answer_split": "### answer:"
}

llama_medqa = Inferer("meta-llama/Llama-2-7b-chat-hf", prompt_template)
response = llama_medqa(choices=)