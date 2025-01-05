"""Cohere tokenizer implementation."""

from typing import List, Optional
import cohere
from .base_tokenizer import BaseTokenizer

class CohereTokenizer(BaseTokenizer):
    """Cohere tokenizer for text encoding and decoding."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Cohere tokenizer with API key."""
        super().__init__()
        self.client = cohere.Client(api_key) if api_key else None
    
    def encode(self, text: str) -> List[int]:
        """Encode text into token IDs."""
        if not self.client:
            return super().encode(text)
        try:
            response = self.client.tokenize(text=text)
            return response.tokens
        except:
            return super().encode(text)
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back into text."""
        if not self.client:
            return super().decode(token_ids)
        try:
            response = self.client.detokenize(tokens=token_ids)
            return response.text
        except:
            return super().decode(token_ids)
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the text."""
        if not self.client:
            return super().count_tokens(text)
        try:
            response = self.client.tokenize(text=text)
            return response.length
        except:
            return super().count_tokens(text)
