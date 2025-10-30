import os
from typing import List

import paddleformers.transformers as transformers


def select_tokenizer(tokenizer_type, tokenizer_path):
    if tokenizer_type == "hf":
        return HFTokenizer(model_path=tokenizer_path)


class HFTokenizer:
    """
    Tokenizer from HF models
    """

    def __init__(self, model_path) -> None:
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)

    def text_to_tokens(self, text: str) -> List[str]:
        tokens = self.tokenizer.tokenize(text)
        return tokens

    def tokens_to_text(self, tokens: List[int]) -> str:
        text = self.tokenizer.convert_tokens_to_string(tokens)
        return text
