"""AI21 Labs API provider integration."""

from typing import Dict, Any, Optional
import ai21
from .provider_template import ProviderTemplate

class AI21Provider(ProviderTemplate):
    """AI21 Labs API provider for text generation and embeddings."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize AI21 provider with API key."""
        self.api_key = api_key
        if api_key:
            ai21.api_key = api_key
    
    def get_model_limits(self, model_name: str) -> Dict[str, Any]:
        """Get the limits for AI21 models."""
        MODEL_LIMITS = {
            "j2-light": {
                "type": "text",
                "token_limit": 8192,
                "max_output_tokens": 2048,
                "additional_constraints": {
                    "temperature_range": [0.0, 1.0],
                    "top_p_range": [0.0, 1.0],
                    "top_k_range": [1, 400],
                    "presence_penalty_range": [-2.0, 2.0],
                    "count_penalty_range": [-2.0, 2.0]
                }
            },
            "j2-mid": {
                "type": "text",
                "token_limit": 8192,
                "max_output_tokens": 2048,
                "additional_constraints": {
                    "temperature_range": [0.0, 1.0],
                    "top_p_range": [0.0, 1.0],
                    "top_k_range": [1, 400],
                    "presence_penalty_range": [-2.0, 2.0],
                    "count_penalty_range": [-2.0, 2.0]
                }
            },
            "j2-ultra": {
                "type": "text",
                "token_limit": 8192,
                "max_output_tokens": 2048,
                "additional_constraints": {
                    "temperature_range": [0.0, 1.0],
                    "top_p_range": [0.0, 1.0],
                    "top_k_range": [1, 400],
                    "presence_penalty_range": [-2.0, 2.0],
                    "count_penalty_range": [-2.0, 2.0]
                }
            },
            "j2-grande-instruct": {
                "type": "text",
                "token_limit": 8192,
                "max_output_tokens": 2048,
                "additional_constraints": {
                    "temperature_range": [0.0, 1.0],
                    "top_p_range": [0.0, 1.0],
                    "top_k_range": [1, 400],
                    "presence_penalty_range": [-2.0, 2.0],
                    "count_penalty_range": [-2.0, 2.0]
                }
            },
            "j2-embed": {
                "type": "embedding",
                "token_limit": 1024,
                "dimensions": 768,
                "additional_constraints": {
                    "batch_size": 32,
                    "pooling": ["first", "mean", "last"]
                }
            }
        }
        
        if not model_name:
            return MODEL_LIMITS
        return MODEL_LIMITS.get(model_name, {})
