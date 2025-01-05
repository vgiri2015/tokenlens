from typing import Dict, Any
from .provider_template import ProviderTemplate

class StanfordProvider(ProviderTemplate):
    def __init__(self):
        super().__init__()
        self.provider_name = "stanford"
        
    def check_text_limits(self, text: str, model: str) -> Dict[str, Any]:
        """Check text limits for Stanford models."""
        config = self._get_model_config(model)
        if not config:
            raise ValueError(f"Model {model} not found for provider {self.provider_name}")
            
        token_limit = config.get("token_limit", 2048)
        max_response_tokens = config.get("max_response_tokens", 1024)
        
        # Use GPT-2 tokenizer for approximation
        token_count = len(text.split())  # Simple approximation
        
        return {
            "token_count": token_count,
            "token_limit": token_limit,
            "max_response_tokens": max_response_tokens,
            "within_limit": token_count <= token_limit,
            "provider": self.provider_name,
            "model": model
        }
        
    def list_models(self) -> Dict[str, Any]:
        """List available Stanford models."""
        return {
            "text_models": ["alpaca"],
            "provider": self.provider_name
        }
