"""Google AI tokenizer implementation."""

from typing import List, Optional
import google.generativeai as genai
from .base_tokenizer import BaseTokenizer

class GoogleTokenizer(BaseTokenizer):
    """Google AI tokenizer for text encoding and decoding."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Google tokenizer with API key."""
        super().__init__()
        if api_key:
            genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro') if api_key else None
    
    def encode(self, text: str) -> List[int]:
        """Encode text into token IDs."""
        if not self.model:
            return super().encode(text)
        try:
            response = self.model.count_tokens(text)
            return list(range(response.total_tokens))  # Google doesn't expose token IDs
        except:
            return super().encode(text)
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back into text."""
        if not self.model:
            return super().decode(token_ids)
        try:
            # Google doesn't provide decoding functionality
            return super().decode(token_ids)
        except:
            return super().decode(token_ids)
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the text."""
        if not self.model:
            return super().count_tokens(text)
        try:
            response = self.model.count_tokens(text)
            return response.total_tokens
        except:
            return super().count_tokens(text)
