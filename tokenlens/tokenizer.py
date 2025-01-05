"""TokenLens core functionality for counting tokens and validating token limits."""

from typing import Optional, Union

from .tokenizers import OpenAITokenizer

__all__ = [
    "OpenAITokenizer",
]

def count_tokens(text: str, provider: str = "openai", model: Optional[str] = None) -> int:
    """Count the number of tokens in the given text using the specified provider's tokenizer.
    
    Args:
        text: The text to count tokens for
        provider: The provider to use for tokenization (e.g. 'openai', 'anthropic', etc.)
        model: Optional model name to use for tokenization
        
    Returns:
        Number of tokens in the text
    """
    if provider == "openai":
        tokenizer = OpenAITokenizer()
    else:
        raise ValueError(f"Provider {provider} not supported or its dependencies are not installed")
        
    return tokenizer.count_tokens(text, model)

def validate_token_limit(text: str, max_tokens: int, provider: str = "openai", model: Optional[str] = None) -> bool:
    """Validate that the text is within the specified token limit.
    
    Args:
        text: The text to validate
        max_tokens: Maximum number of tokens allowed
        provider: The provider to use for tokenization
        model: Optional model name to use for tokenization
        
    Returns:
        True if text is within token limit, False otherwise
    """
    token_count = count_tokens(text, provider, model)
    return token_count <= max_tokens
