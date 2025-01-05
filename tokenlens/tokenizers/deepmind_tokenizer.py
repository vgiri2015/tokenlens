"""DeepMind tokenizer implementation."""

from typing import List, Optional
import requests
from .base_tokenizer import BaseTokenizer

class DeepMindTokenizer(BaseTokenizer):
    """DeepMind tokenizer for text encoding and decoding."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize DeepMind tokenizer with API key."""
        super().__init__()
        self.api_key = api_key
        self.base_url = "https://api.deepmind.com/v1"
        if api_key:
            self.headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
    
    def encode(self, text: str) -> List[int]:
        """Encode text into token IDs."""
        if not self.api_key:
            return super().encode(text)
        try:
            response = requests.post(
                f"{self.base_url}/tokenize",
                headers=self.headers,
                json={"text": text}
            )
            if response.status_code == 200:
                return response.json()["tokens"]
            return super().encode(text)
        except:
            return super().encode(text)
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back into text."""
        if not self.api_key:
            return super().decode(token_ids)
        try:
            response = requests.post(
                f"{self.base_url}/detokenize",
                headers=self.headers,
                json={"tokens": token_ids}
            )
            if response.status_code == 200:
                return response.json()["text"]
            return super().decode(token_ids)
        except:
            return super().decode(token_ids)
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the text."""
        if not self.api_key:
            return super().count_tokens(text)
        try:
            response = requests.post(
                f"{self.base_url}/count_tokens",
                headers=self.headers,
                json={"text": text}
            )
            if response.status_code == 200:
                return response.json()["count"]
            return super().count_tokens(text)
        except:
            return super().count_tokens(text)
