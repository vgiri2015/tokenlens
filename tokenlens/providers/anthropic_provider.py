"""Anthropic API provider integration."""

from typing import Dict, Any, Optional
import anthropic
from .provider_template import ProviderTemplate

class AnthropicProvider(ProviderTemplate):
    """Anthropic API provider for text generation and vision."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Anthropic provider with API key."""
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else None
    
    def get_model_limits(self, model_name: str) -> Dict[str, Any]:
        """Get the limits for Anthropic models."""
        MODEL_LIMITS = {
            "claude-3-opus": {
                "type": "text",
                "token_limit": 200000,
                "max_output_tokens": 4096,
                "additional_constraints": {
                    "vision_support": True,
                    "max_image_size": "100MB",
                    "supported_image_types": ["png", "jpeg", "gif", "webp"],
                    "temperature_range": [0.0, 1.0],
                    "top_p_range": [0.0, 1.0],
                    "top_k_range": [1, 500]
                }
            },
            "claude-3-sonnet": {
                "type": "text",
                "token_limit": 200000,
                "max_output_tokens": 4096,
                "additional_constraints": {
                    "vision_support": True,
                    "max_image_size": "100MB",
                    "supported_image_types": ["png", "jpeg", "gif", "webp"],
                    "temperature_range": [0.0, 1.0],
                    "top_p_range": [0.0, 1.0],
                    "top_k_range": [1, 500]
                }
            },
            "claude-2.1": {
                "type": "text",
                "token_limit": 200000,
                "max_output_tokens": 4096,
                "additional_constraints": {
                    "vision_support": False,
                    "temperature_range": [0.0, 1.0],
                    "top_p_range": [0.0, 1.0],
                    "top_k_range": [1, 500]
                }
            },
            "claude-2.0": {
                "type": "text",
                "token_limit": 100000,
                "max_output_tokens": 4096,
                "additional_constraints": {
                    "vision_support": False,
                    "temperature_range": [0.0, 1.0],
                    "top_p_range": [0.0, 1.0],
                    "top_k_range": [1, 500]
                }
            },
            "claude-instant-1.2": {
                "type": "text",
                "token_limit": 100000,
                "max_output_tokens": 4096,
                "additional_constraints": {
                    "vision_support": False,
                    "temperature_range": [0.0, 1.0],
                    "top_p_range": [0.0, 1.0],
                    "top_k_range": [1, 500]
                }
            }
        }
        
        if not model_name:
            return MODEL_LIMITS
        return MODEL_LIMITS.get(model_name, {})
        
    def _count_tokens(self, content: str) -> int:
        """Count tokens using Anthropic's tokenizer."""
        if self.client:
            return self.client.count_tokens(content)
        return super()._count_tokens(content)  # Fallback to word-based counting
