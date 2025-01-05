"""Base tokenizer implementation."""

from abc import ABC, abstractmethod
from typing import List, Optional

class BaseTokenizer(ABC):
    """Base class for all tokenizers."""
    
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Encode text into tokens."""
        pass
    
    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        """Decode tokens back into text."""
        pass
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the text."""
        return len(self.encode(text))
