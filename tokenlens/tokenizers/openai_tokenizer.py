"""OpenAI tokenizer implementation."""

from typing import List, Optional
import tiktoken
from openai import OpenAI
from .base import BaseTokenizer

class OpenAITokenizer(BaseTokenizer):
    """OpenAI tokenizer for text encoding and decoding."""
    
    def __init__(self, model_name: str = "gpt-4", api_key: Optional[str] = None):
        """Initialize OpenAI tokenizer with model name and API key."""
        self.model_name = model_name
        try:
            self.client = OpenAI(api_key=api_key) if api_key else None
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except Exception as e:
            raise ValueError(f"Failed to initialize OpenAI tokenizer: {str(e)}")
    
    def encode(self, text: str) -> List[int]:
        """Encode text into token IDs."""
        if not self.tokenizer:
            raise ValueError("Tokenizer not initialized")
        return self.tokenizer.encode(text)
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back into text."""
        if not self.tokenizer:
            raise ValueError("Tokenizer not initialized")
        return self.tokenizer.decode(token_ids)
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the text."""
        if not self.tokenizer:
            raise ValueError("Tokenizer not initialized")
        return len(self.tokenizer.encode(text))
