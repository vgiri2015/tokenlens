"""Tokenizer factory for creating tokenizer instances."""

from typing import Dict, Type, Optional
from importlib import import_module
from .base import BaseTokenizer

class TokenizerFactory:
    """Factory class for creating tokenizer instances."""
    
    _tokenizers: Dict[str, str] = {
        # Text Generation
        "openai": "openai_tokenizer.OpenAITokenizer",
        "anthropic": "anthropic_tokenizer.AnthropicTokenizer",
        "google": "google_tokenizer.GoogleTokenizer",
        "huggingface": "huggingface_tokenizer.HuggingFaceTokenizer",
        "meta": "meta_tokenizer.MetaTokenizer",
        "mistral": "mistral_tokenizer.MistralTokenizer",
        "cohere": "cohere_tokenizer.CohereTokenizer",
        "ai21": "ai21_tokenizer.AI21Tokenizer",
        "qwen": "qwen_tokenizer.QwenTokenizer",
        "stanford": "stanford_tokenizer.StanfordTokenizer",
        "deepmind": "deepmind_tokenizer.DeepMindTokenizer",
        
        # Provider Aliases
        "meta.llama2": "meta_tokenizer.MetaTokenizer",
        "anthropic.claude": "anthropic_tokenizer.AnthropicTokenizer",
        "google.palm": "google_tokenizer.GoogleTokenizer",
        
        # Image/Video/Avatar providers (no tokenizers needed)
        "stability": None,
        "midjourney": None,
        "adobe": None,
        "ideogram": None,
        "runway": None,
        "nightcafe": None,
        "openart": None,
        "synthesia": None,
        "d-id": None,
        "replika": None,
        "realm": None,
        "starrytars": None,
        "fugatto": None,
        
        # Provider aliases
        "amazon.titan": "openai_tokenizer.OpenAITokenizer",  # Uses GPT-2 tokenizer
        "microsoft.azure": "openai_tokenizer.OpenAITokenizer",  # Uses tiktoken
    }
    
    @classmethod
    def get_tokenizer(cls, tokenizer_name: str) -> Optional[Type[BaseTokenizer]]:
        """Get a tokenizer instance by name."""
        if tokenizer_name not in cls._tokenizers:
            return None
            
        module_path = cls._tokenizers[tokenizer_name]
        if module_path is None:
            return None
        
        module_name, class_name = module_path.rsplit(".", 1)
        
        try:
            module = import_module(f".{module_name}", package="tokenlens.tokenizers")
            return getattr(module, class_name)
        except ImportError:
            return None
    
    @classmethod
    def register_tokenizer(cls, name: str, tokenizer_path: str) -> None:
        """Register a new tokenizer."""
        cls._tokenizers[name] = tokenizer_path
    
    @classmethod
    def get_supported_tokenizers(cls) -> list[str]:
        """Get list of supported tokenizers."""
        return [p for p, t in cls._tokenizers.items() if t is not None]
