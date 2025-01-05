"""Mistral AI provider integration."""

from typing import Dict, Any, Optional
import mistralai
from .provider_template import ProviderTemplate

class MistralProvider(ProviderTemplate):
    """Mistral AI provider for text generation and embeddings."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Mistral provider with API key."""
        self.client = mistralai.MistralClient(api_key=api_key) if api_key else None
    
    def get_model_limits(self, model_name: str) -> Dict[str, Any]:
        """Get the limits for Mistral models."""
        MODEL_LIMITS = {
            "mistral-tiny": {
                "type": "text",
                "token_limit": 32768,
                "max_output_tokens": 8192,
                "additional_constraints": {
                    "temperature_range": [0.0, 1.0],
                    "top_p_range": [0.0, 1.0],
                    "presence_penalty_range": [-2.0, 2.0],
                    "frequency_penalty_range": [-2.0, 2.0],
                    "system_prompt": True,
                    "function_calling": True,
                    "json_mode": True
                }
            },
            "mistral-small": {
                "type": "text",
                "token_limit": 32768,
                "max_output_tokens": 8192,
                "additional_constraints": {
                    "temperature_range": [0.0, 1.0],
                    "top_p_range": [0.0, 1.0],
                    "presence_penalty_range": [-2.0, 2.0],
                    "frequency_penalty_range": [-2.0, 2.0],
                    "system_prompt": True,
                    "function_calling": True,
                    "json_mode": True
                }
            },
            "mistral-medium": {
                "type": "text",
                "token_limit": 32768,
                "max_output_tokens": 8192,
                "additional_constraints": {
                    "temperature_range": [0.0, 1.0],
                    "top_p_range": [0.0, 1.0],
                    "presence_penalty_range": [-2.0, 2.0],
                    "frequency_penalty_range": [-2.0, 2.0],
                    "system_prompt": True,
                    "function_calling": True,
                    "json_mode": True
                }
            },
            "mistral-large": {
                "type": "text",
                "token_limit": 32768,
                "max_output_tokens": 8192,
                "additional_constraints": {
                    "temperature_range": [0.0, 1.0],
                    "top_p_range": [0.0, 1.0],
                    "presence_penalty_range": [-2.0, 2.0],
                    "frequency_penalty_range": [-2.0, 2.0],
                    "system_prompt": True,
                    "function_calling": True,
                    "json_mode": True
                }
            },
            "mistral-embed": {
                "type": "embedding",
                "token_limit": 32768,
                "dimensions": 1024,
                "additional_constraints": {
                    "batch_size": 96,
                    "encoding": "cl100k_base",
                    "normalize": True
                }
            }
        }
        
        if not model_name:
            return MODEL_LIMITS
        return MODEL_LIMITS.get(model_name, {})
        
    def _count_tokens(self, content: str) -> int:
        """Count tokens using Mistral's tokenizer."""
        if self.client:
            try:
                return self.client.count_tokens(content)
            except:
                pass
        return super()._count_tokens(content)  # Fallback to word-based counting
