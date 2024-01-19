import json
import torch
import logging
from typing import Dict, Optional

class DataHandler:
    """Helper class to handle prompt generation and data tokenization.

    Args:
        tokenizer: The tokenizer to use for tokenization.
        prompt_template (str, optional):
            The path to the JSON file containing the prompt template.
            Defaults to "prompts/medalpaca.json".
        model_max_length (int, optional):
            The maximum length of the tokenized sequence.
            Should not exceed 2048, as LLaMA is trained with this. Defaults to 256.
        train_on_inputs (bool, optional):
            If False, masks out inputs in loss. Defaults to True.

    Methods:
        tokenize(prompt: str, add_eos_token: bool = True) -> Dict:
            Tokenizes the given prompt and optionally adds an end-of-sequence (EOS) token.

        generate_and_tokenize_prompt(data_point: Dict) -> Dict:
            Generates a prompt based on the given data point and tokenizes it.

    """

    def __init__(
            self,
            tokenizer,
            prompt_template: dict,
            model_max_length: int = 256,
            train_on_inputs: bool = True,
    ) -> None:
        self.prompt_template = prompt_template
        self.model_max_length = model_max_length
        self.train_on_inputs = train_on_inputs
        self.tokenizer = tokenizer

    def tokenize(self, prompt: str, add_eos_token: bool = True, return_tensors: str = None, truncation: bool = True) -> \
    Dict[str, list]:
        """
        Tokenize the given prompt and optionally add an end-of-sequence (EOS) token.

        This function tokenizes the input prompt without adding special tokens by default.
        If the `add_eos_token` parameter is True and the tokenized sequence doesn't already
        end with an EOS token, an EOS token will be added to the end of the sequence.

        Args:
            prompt (str): The text to be tokenized.
            add_eos_token (bool, optional): Whether to add an EOS token at the end of
                the tokenized sequence. Defaults to True.
            return_tensors (str, optional): If tensors should be returned (and what type).
            trunctaion (bool, optional); Whether to truncate the input to max_model_length


        Returns:
            Dict: A dictionary containing the tokenized data:
                - input_ids: The tokenized input IDs of the prompt.
                - attention_mask: The attention mask for the tokenized input IDs.
                - labels: The labels for the tokenized input IDs (identical to input_ids).
        """
        # TODO: optimize (roll back changes from debugging)
        result: Dict = self.tokenizer(
            prompt,
            truncation=truncation,
            max_length=self.model_max_length,
            padding=False,
            return_tensors=return_tensors,
            add_special_tokens=False,
        )
        if (
                result["input_ids"][-1] != self.tokenizer.eos_token_id
                and len(result["input_ids"]) < self.model_max_length
                and add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(self, data_point: Dict):
        """
        Generate a prompt based on the given data point and tokenize it.

        This function creates a prompt using the given data point, which consists
        of an instruction, input, and output. It then tokenizes the generated prompt
        and returns the tokenized representation. If the `train_on_inputs` global
        variable is False, the function will create a user prompt without the
        expected output and only tokenize that part, masking the output part in the
        "labels" field with -100.

        Args:
            data_point (Dict): A dictionary containing the following keys:
                - instruction: The instruction text for the prompt.
                - input: The input text for the prompt.
                - output: The output text for the prompt.

        Returns:
            Dict: A dictionary containing the tokenized prompt and associated data:
                - input_ids: The tokenized input IDs of the generated prompt.
                - attention_mask: The attention mask for the tokenized input IDs.
                - labels: The labels to be used during model training, with the output
                part unmasked and the rest masked with -100 if `train_on_inputs` is False.
        """
        prompt: str = self.generate_prompt(
            question=data_point.get("question", ""),
            choices=data_point.get("choices", ""),
            answer=data_point.get("answer", ""),
        )
        tokenized_prompt: Dict = self.tokenize(prompt)
        if not self.train_on_inputs:
            user_prompt: str = self.generate_prompt(
                question=data_point.get("question", ""), choices=data_point.get("choices", "")
            )
            tokenized_user_prompt: Dict = self.tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            # mask out the inputs
            tokenized_prompt["labels"] = [
                -100 if i < user_prompt_len else label
                for i, label in enumerate(tokenized_prompt["labels"])
            ]
        return tokenized_prompt

    def generate_prompt(
            self,
            question: Optional[str] = None,
            choices: Optional[str] = None,
            answer: Optional[str] = None,
    ) -> str:

        if not any([question, choices, answer]):
            raise ValueError("At least one of `instruction`, `input`, `output` should be defined")

        prompt = (
            f'{self.prompt_template["primer"]}'
            f'{self.prompt_template["question"]}{question or ""}'
            f'{self.prompt_template["choices"]}{choices or ""}'
            f'{self.prompt_template["answer"]}{answer or ""}'
        )

        return prompt

    def resolve_output(self, output: str):
        pass