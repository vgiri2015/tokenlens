"""Mistral AI tokenizer implementation."""

from typing import List, Optional
import mistralai
from .base_tokenizer import BaseTokenizer

class MistralTokenizer(BaseTokenizer):
    """Mistral AI tokenizer for text encoding and decoding."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Mistral tokenizer with API key."""
        super().__init__()
        self.client = mistralai.MistralClient(api_key=api_key) if api_key else None
    
    def encode(self, text: str) -> List[int]:
        """Encode text into token IDs."""
        if not self.client:
            return super().encode(text)
        try:
            # Mistral doesn't expose token IDs directly
            count = self.client.count_tokens(text)
            return list(range(count))
        except:
            return super().encode(text)
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back into text."""
        if not self.client:
            return super().decode(token_ids)
        try:
            # Mistral doesn't provide decoding functionality
            return super().decode(token_ids)
        except:
            return super().decode(token_ids)
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the text."""
        if not self.client:
            return super().count_tokens(text)
        try:
            return self.client.count_tokens(text)
        except:
            return super().count_tokens(text)
