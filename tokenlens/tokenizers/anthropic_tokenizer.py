"""Anthropic tokenizer implementation."""

from typing import List, Optional
import anthropic
from .base import BaseTokenizer

class AnthropicTokenizer(BaseTokenizer):
    """Anthropic tokenizer for text encoding and decoding."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Anthropic tokenizer with API key."""
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else None
    
    def encode(self, text: str) -> List[int]:
        """Encode text into token IDs."""
        if not self.client:
            raise ValueError("Tokenizer not initialized")
        try:
            return self.client.count_tokens(text)
        except Exception as e:
            raise ValueError(f"Failed to encode text: {str(e)}")
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back into text."""
        raise NotImplementedError("Anthropic does not support token decoding")
