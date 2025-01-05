"""Midjourney API provider integration."""

from typing import Dict, Any, Optional
import requests
from .provider_template import ProviderTemplate

class MidjourneyProvider(ProviderTemplate):
    """Midjourney provider for image generation."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Midjourney provider with API key."""
        self.api_key = api_key
        self.base_url = "https://api.midjourney.com/v1"
        if api_key:
            self.headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
    
    def get_model_limits(self, model_name: str) -> Dict[str, Any]:
        """Get the limits for Midjourney models."""
        MODEL_LIMITS = {
            "midjourney-v6": {
                "type": "image",
                "max_resolution": "1024x1024",
                "supported_formats": ["png", "jpeg", "webp"],
                "additional_constraints": {
                    "prompt_length": 6000,
                    "negative_prompt": True,
                    "styles": [
                        "raw", "steampunk", "synthwave", "cyberpunk",
                        "anime", "manga", "fantasy", "medieval", "sci-fi",
                        "abstract", "realistic", "cinematic", "3d-model",
                        "pixel-art", "vector", "studio-photo", "painting"
                    ],
                    "aspect_ratios": [
                        "1:1", "4:3", "3:4", "16:9", "9:16", "2:1", "1:2"
                    ],
                    "quality_options": ["draft", "regular", "max"],
                    "style_versions": [1, 2, 3, 4, 5, 6],
                    "chaos_range": [0, 100],
                    "stylize_range": [0, 1000],
                    "weird_range": [0, 3000],
                    "tile": True,
                    "upscale": True,
                    "vary": True,
                    "pan": True,
                    "zoom": True,
                    "remaster": True
                }
            },
            "niji-v6": {
                "type": "image",
                "max_resolution": "1024x1024",
                "supported_formats": ["png", "jpeg", "webp"],
                "additional_constraints": {
                    "prompt_length": 6000,
                    "negative_prompt": True,
                    "styles": [
                        "anime", "manga", "kawaii", "chibi", "mecha",
                        "pixel-art", "watercolor", "ink", "line-art"
                    ],
                    "aspect_ratios": [
                        "1:1", "4:3", "3:4", "16:9", "9:16", "2:1", "1:2"
                    ],
                    "quality_options": ["draft", "regular", "max"],
                    "style_versions": [1, 2, 3, 4, 5, 6],
                    "chaos_range": [0, 100],
                    "stylize_range": [0, 1000],
                    "weird_range": [0, 3000],
                    "tile": True,
                    "upscale": True,
                    "vary": True,
                    "pan": True,
                    "zoom": True,
                    "remaster": True
                }
            },
            "midjourney-turbo": {
                "type": "image",
                "max_resolution": "1024x1024",
                "supported_formats": ["png", "jpeg", "webp"],
                "additional_constraints": {
                    "prompt_length": 6000,
                    "negative_prompt": True,
                    "styles": [
                        "raw", "steampunk", "synthwave", "cyberpunk",
                        "anime", "manga", "fantasy", "medieval", "sci-fi",
                        "abstract", "realistic", "cinematic", "3d-model",
                        "pixel-art", "vector", "studio-photo", "painting"
                    ],
                    "aspect_ratios": [
                        "1:1", "4:3", "3:4", "16:9", "9:16", "2:1", "1:2"
                    ],
                    "quality_options": ["draft", "regular"],
                    "style_versions": [1, 2, 3, 4, 5, 6],
                    "chaos_range": [0, 100],
                    "stylize_range": [0, 1000],
                    "weird_range": [0, 3000],
                    "tile": True,
                    "upscale": True,
                    "vary": True,
                    "pan": True,
                    "zoom": True,
                    "remaster": True,
                    "fast_mode": True
                }
            }
        }
        
        if not model_name:
            return MODEL_LIMITS
        return MODEL_LIMITS.get(model_name, {})
