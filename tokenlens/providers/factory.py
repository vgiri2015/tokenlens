from typing import Dict, Type
from . import BaseProvider
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider
from .huggingface_provider import HuggingFaceProvider
from .google_provider import GoogleProvider
from .stability_provider import StabilityProvider
from .meta_provider import MetaProvider
from .qwen_provider import QwenProvider
from .mistral_provider import MistralProvider
from .stanford_provider import StanfordProvider
from .deepmind_provider import DeepMindProvider
from .midjourney_provider import MidjourneyProvider
from .adobe_provider import AdobeProvider
from .ideogram_provider import IdeogramProvider
from .runway_provider import RunwayProvider
from .microsoft_provider import MicrosoftProvider
from .synthesia_provider import SynthesiaProvider
from .did_provider import DIDProvider
from .replika_provider import ReplikaProvider
from .nightcafe_provider import NightCafeProvider
from .openart_provider import OpenArtProvider
from .realm_provider import RealmProvider
from .starrytars_provider import StarryTarsProvider
from .fugatto_provider import FugattoProvider

class ProviderFactory:
    """Factory class for creating provider API clients."""
    
    _providers: Dict[str, Type[BaseProvider]] = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "huggingface": HuggingFaceProvider,
        "google": GoogleProvider,
        "stability": StabilityProvider,
        "meta": MetaProvider,
        "qwen": QwenProvider,
        "mistral": MistralProvider,
        "stanford": StanfordProvider,
        "deepmind": DeepMindProvider,
        "midjourney": MidjourneyProvider,
        "adobe": AdobeProvider,
        "ideogram": IdeogramProvider,
        "runway": RunwayProvider,
        "microsoft": MicrosoftProvider,
        "synthesia": SynthesiaProvider,
        "d-id": DIDProvider,
        "replika": ReplikaProvider,
        "nightcafe": NightCafeProvider,
        "openart": OpenArtProvider,
        "realm": RealmProvider,
        "starrytars": StarryTarsProvider,
        "fugatto": FugattoProvider,
    }
    
    @classmethod
    def get_provider(cls, provider_name: str, api_key: str = None) -> BaseProvider:
        """Get a provider instance for the specified provider."""
        provider_class = cls._providers.get(provider_name.lower())
        if not provider_class:
            raise ValueError(f"No provider implementation available for: {provider_name}")
        
        return provider_class(api_key)
    
    @classmethod
    def register_provider(cls, provider_name: str, provider_class: Type[BaseProvider]):
        """Register a new provider implementation."""
        cls._providers[provider_name.lower()] = provider_class
