"""AI21 tokenizer implementation."""

from typing import List, Optional
import ai21
from .base_tokenizer import BaseTokenizer

class AI21Tokenizer(BaseTokenizer):
    """AI21 tokenizer for text encoding and decoding."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize AI21 tokenizer with API key."""
        super().__init__()
        self.api_key = api_key
        if api_key:
            ai21.api_key = api_key
    
    def encode(self, text: str) -> List[int]:
        """Encode text into token IDs."""
        if not self.api_key:
            return super().encode(text)
        try:
            response = ai21.TokenizeRequest(text=text)
            return [token.id for token in response.tokens]
        except:
            return super().encode(text)
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back into text."""
        if not self.api_key:
            return super().decode(token_ids)
        try:
            response = ai21.DetokenizeRequest(token_ids=token_ids)
            return response.text
        except:
            return super().decode(token_ids)
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the text."""
        if not self.api_key:
            return super().count_tokens(text)
        try:
            response = ai21.TokenizeRequest(text=text)
            return len(response.tokens)
        except:
            return super().count_tokens(text)
