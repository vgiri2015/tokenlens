"""Provider factory for creating provider instances."""

from typing import Dict, Type, Optional
from importlib import import_module
from .provider_template import ProviderTemplate

class ProviderFactory:
    """Factory class for creating provider instances."""
    
    _provider_modules = {
        "openai": "openai_provider.OpenAIProvider",
        "anthropic": "anthropic_provider.AnthropicProvider",
        "google": "google_provider.GoogleProvider",
        "huggingface": "huggingface_provider.HuggingFaceProvider",
        "meta": "meta_provider.MetaProvider",
        "mistral": "mistral_provider.MistralProvider",
        "cohere": "cohere_provider.CohereProvider",
        "ai21": "ai21_provider.AI21Provider",
        "qwen": "qwen_provider.QwenProvider",
        "stanford": "stanford_provider.StanfordProvider",
        "deepmind": "deepmind_provider.DeepMindProvider"
    }

    @classmethod
    def get_provider(cls, provider_name: str) -> Optional[Type[ProviderTemplate]]:
        """Get a provider instance by name.
        
        Args:
            provider_name: Name of the provider to get
            
        Returns:
            Provider class if found and successfully imported, None otherwise
        """
        # Get base provider name (e.g., "openai" from "openai.gpt-4")
        base_provider = provider_name.split('.')[0]
        
        if base_provider not in cls._provider_modules:
            return None
            
        try:
            # Import the provider module only when needed
            module_path = cls._provider_modules[base_provider]
            module_name, class_name = module_path.rsplit('.', 1)
            
            # Import the specific provider module
            module = import_module(f".{module_name}", package="tokenlens.providers")
            return getattr(module, class_name)
        except ImportError:
            # If import fails (e.g., missing dependencies), return None
            return None

    @classmethod
    def get_supported_providers(cls) -> list[str]:
        """Get list of supported providers."""
        return list(cls._provider_modules.keys())
