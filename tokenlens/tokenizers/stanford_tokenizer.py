"""Stanford AI tokenizer implementation."""

from typing import List, Optional
import requests
from transformers import AutoTokenizer
from .base_tokenizer import BaseTokenizer

class StanfordTokenizer(BaseTokenizer):
    """Stanford AI tokenizer for text encoding and decoding."""
    
    def __init__(self, model_name: str = "stanford-alpaca/alpaca-7b", api_key: Optional[str] = None):
        """Initialize Stanford tokenizer with model name and API key."""
        super().__init__()
        self.model_name = model_name
        self.api_key = api_key
        try:
            # Try to load from HuggingFace
            auth_token = {"use_auth_token": api_key} if api_key else {}
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, **auth_token)
        except:
            self.tokenizer = None
    
    def encode(self, text: str) -> List[int]:
        """Encode text into token IDs."""
        if not self.tokenizer:
            return super().encode(text)
        try:
            return self.tokenizer.encode(text)
        except:
            return super().encode(text)
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back into text."""
        if not self.tokenizer:
            return super().decode(token_ids)
        try:
            return self.tokenizer.decode(token_ids)
        except:
            return super().decode(token_ids)
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the text."""
        if not self.tokenizer:
            return super().count_tokens(text)
        try:
            return len(self.tokenizer.encode(text))
        except:
            return super().count_tokens(text)
