"""Meta AI provider integration."""

from typing import Dict, Any, Optional
import requests
from .provider_template import ProviderTemplate

class MetaProvider(ProviderTemplate):
    """Meta AI provider for text and image generation."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Meta provider with API key."""
        self.api_key = api_key
        self.base_url = "https://api.meta.ai/v1"
        if api_key:
            self.headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
    
    def get_model_limits(self, model_name: str) -> Dict[str, Any]:
        """Get the limits for Meta models."""
        MODEL_LIMITS = {
            # Text Models - Llama family
            "llama-2-70b-chat": {
                "type": "text",
                "token_limit": 4096,
                "max_output_tokens": 1024,
                "additional_constraints": {
                    "temperature_range": [0.0, 2.0],
                    "top_p_range": [0.0, 1.0],
                    "top_k_range": [1, 100],
                    "repetition_penalty_range": [1.0, 2.0],
                    "system_prompt": True,
                    "function_calling": True
                }
            },
            "llama-2-13b-chat": {
                "type": "text",
                "token_limit": 4096,
                "max_output_tokens": 1024,
                "additional_constraints": {
                    "temperature_range": [0.0, 2.0],
                    "top_p_range": [0.0, 1.0],
                    "top_k_range": [1, 100],
                    "repetition_penalty_range": [1.0, 2.0],
                    "system_prompt": True,
                    "function_calling": True
                }
            },
            "llama-2-7b-chat": {
                "type": "text",
                "token_limit": 4096,
                "max_output_tokens": 1024,
                "additional_constraints": {
                    "temperature_range": [0.0, 2.0],
                    "top_p_range": [0.0, 1.0],
                    "top_k_range": [1, 100],
                    "repetition_penalty_range": [1.0, 2.0],
                    "system_prompt": True,
                    "function_calling": True
                }
            },
            
            # Code Models
            "code-llama-34b": {
                "type": "text",
                "token_limit": 8192,
                "max_output_tokens": 2048,
                "additional_constraints": {
                    "temperature_range": [0.0, 2.0],
                    "top_p_range": [0.0, 1.0],
                    "top_k_range": [1, 100],
                    "repetition_penalty_range": [1.0, 2.0],
                    "language_support": [
                        "python", "javascript", "java", "cpp", "go",
                        "php", "ruby", "rust", "typescript", "swift"
                    ],
                    "infilling": True,
                    "code_completion": True
                }
            },
            
            # Image Models
            "imagebind": {
                "type": "image",
                "max_resolution": "1024x1024",
                "supported_formats": ["png", "jpeg", "webp"],
                "additional_constraints": {
                    "modalities": ["image", "text", "audio", "video"],
                    "cross_modal": True,
                    "embedding_dim": 1024,
                    "batch_processing": True
                }
            }
        }
        
        if not model_name:
            return MODEL_LIMITS
        return MODEL_LIMITS.get(model_name, {})
    
    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """List all available Meta AI models and their limits."""
        try:
            # Return static list of models since Meta AI doesn't have a public list models API
            models = {}
            for model_name in self.get_model_limits("").keys():
                limits = self.get_model_limits(model_name)
                if limits:
                    models[model_name] = limits
            return models
        except Exception as e:
            return {}
