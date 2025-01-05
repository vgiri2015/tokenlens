"""Amazon Bedrock API provider integration."""

from typing import Dict, Any, Optional
import boto3
from .provider_template import ProviderTemplate

class AmazonProvider(ProviderTemplate):
    """Amazon Bedrock API provider for text and image generation."""
    
    def __init__(self, api_key: Optional[str] = None, region: str = "us-east-1"):
        """Initialize Amazon provider with credentials."""
        self.region = region
        if api_key:
            self.client = boto3.client(
                'bedrock-runtime',
                region_name=region,
                aws_access_key_id=api_key.split(':')[0],
                aws_secret_access_key=api_key.split(':')[1]
            )
    
    def get_model_limits(self, model_name: str) -> Dict[str, Any]:
        """Get the limits for Amazon Bedrock models."""
        MODEL_LIMITS = {
            # Amazon's own models
            "amazon.titan-text-lite-v1": {
                "type": "text",
                "token_limit": 4096,
                "max_output_tokens": 1024,
                "additional_constraints": {
                    "temperature_range": [0.0, 1.0],
                    "top_p_range": [0.0, 1.0]
                }
            },
            "amazon.titan-text-express-v1": {
                "type": "text",
                "token_limit": 8192,
                "max_output_tokens": 2048,
                "additional_constraints": {
                    "temperature_range": [0.0, 1.0],
                    "top_p_range": [0.0, 1.0]
                }
            },
            "amazon.titan-image-generator-v1": {
                "type": "image",
                "max_resolution": "1024x1024",
                "supported_formats": ["png", "jpeg"],
                "additional_constraints": {
                    "quality": ["standard", "premium"],
                    "styles": ["natural", "vivid"],
                    "negative_prompts": True
                }
            },
            
            # Anthropic models on Bedrock
            "anthropic.claude-v2": {
                "type": "text",
                "token_limit": 100000,
                "max_output_tokens": 25000,
                "additional_constraints": {
                    "temperature_range": [0.0, 1.0],
                    "top_p_range": [0.0, 1.0],
                    "top_k_range": [1, 500]
                }
            },
            "anthropic.claude-instant-v1": {
                "type": "text",
                "token_limit": 100000,
                "max_output_tokens": 25000,
                "additional_constraints": {
                    "temperature_range": [0.0, 1.0],
                    "top_p_range": [0.0, 1.0],
                    "top_k_range": [1, 500]
                }
            },
            
            # Meta models on Bedrock
            "meta.llama2-13b-chat-v1": {
                "type": "text",
                "token_limit": 4096,
                "max_output_tokens": 1024,
                "additional_constraints": {
                    "temperature_range": [0.0, 1.0],
                    "top_p_range": [0.0, 1.0]
                }
            },
            "meta.llama2-70b-chat-v1": {
                "type": "text",
                "token_limit": 4096,
                "max_output_tokens": 1024,
                "additional_constraints": {
                    "temperature_range": [0.0, 1.0],
                    "top_p_range": [0.0, 1.0]
                }
            },
            
            # Stability AI models on Bedrock
            "stability.stable-diffusion-xl-v1": {
                "type": "image",
                "max_resolution": "1024x1024",
                "supported_formats": ["png", "jpeg"],
                "additional_constraints": {
                    "styles": ["photographic", "artistic", "digital-art"],
                    "negative_prompts": True,
                    "seed_support": True
                }
            }
        }
        
        if not model_name:
            return MODEL_LIMITS
        return MODEL_LIMITS.get(model_name, {})
