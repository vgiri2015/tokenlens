"""Google AI provider integration."""

from typing import Dict, Any, Optional
import google.generativeai as genai
from .provider_template import ProviderTemplate

class GoogleProvider(ProviderTemplate):
    """Google AI provider for text, image, and video generation."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Google provider with API key."""
        if api_key:
            genai.configure(api_key=api_key)
        self.api_key = api_key
    
    def get_model_limits(self, model_name: str) -> Dict[str, Any]:
        """Get the limits for Google models."""
        MODEL_LIMITS = {
            # Text Models
            "gemini-pro": {
                "type": "text",
                "token_limit": 32768,
                "max_output_tokens": 2048,
                "additional_constraints": {
                    "vision_support": False,
                    "temperature_range": [0.0, 1.0],
                    "top_p_range": [0.0, 1.0],
                    "top_k_range": [1, 40],
                    "stop_sequences": True,
                    "safety_settings": True
                }
            },
            "gemini-pro-vision": {
                "type": "text",
                "token_limit": 32768,
                "max_output_tokens": 2048,
                "additional_constraints": {
                    "vision_support": True,
                    "supported_image_types": ["png", "jpeg", "webp", "heic", "heif"],
                    "max_images": 16,
                    "temperature_range": [0.0, 1.0],
                    "top_p_range": [0.0, 1.0],
                    "top_k_range": [1, 40],
                    "safety_settings": True
                }
            },
            "gemini-ultra": {
                "type": "text",
                "token_limit": 128000,
                "max_output_tokens": 4096,
                "additional_constraints": {
                    "vision_support": True,
                    "supported_image_types": ["png", "jpeg", "webp", "heic", "heif"],
                    "max_images": 16,
                    "temperature_range": [0.0, 1.0],
                    "top_p_range": [0.0, 1.0],
                    "top_k_range": [1, 40],
                    "safety_settings": True
                }
            },
            
            # Image Models
            "imagen-2": {
                "type": "image",
                "max_resolution": "1024x1024",
                "supported_formats": ["png", "jpeg"],
                "additional_constraints": {
                    "photorealism": True,
                    "styles": ["natural", "vivid"],
                    "negative_prompts": True,
                    "safety_settings": True
                }
            },
            
            # Video Models
            "video-palm": {
                "type": "video",
                "max_duration": 60,  # seconds
                "supported_formats": ["mp4"],
                "additional_constraints": {
                    "max_resolution": "1080p",
                    "fps_range": [24, 60],
                    "safety_settings": True
                }
            }
        }
        
        if not model_name:
            return MODEL_LIMITS
        return MODEL_LIMITS.get(model_name, {})
        
    def _count_tokens(self, content: str) -> int:
        """Count tokens using Google's tokenizer."""
        if self.api_key:
            try:
                model = genai.GenerativeModel('gemini-pro')
                return model.count_tokens(content).total_tokens
            except:
                pass
        return super()._count_tokens(content)  # Fallback to word-based counting
