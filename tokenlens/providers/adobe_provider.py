"""Adobe Firefly API provider integration."""

from typing import Dict, Any, Optional
import requests
from .provider_template import ProviderTemplate

class AdobeProvider(ProviderTemplate):
    """Adobe Firefly API provider for image generation."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Adobe provider with API key."""
        self.api_key = api_key
        self.base_url = "https://firefly-api.adobe.io/v2"
        if api_key:
            self.headers = {
                "x-api-key": api_key,
                "Content-Type": "application/json"
            }
    
    def get_model_limits(self, model_name: str) -> Dict[str, Any]:
        """Get the limits for Adobe Firefly models."""
        MODEL_LIMITS = {
            "firefly-v2": {
                "type": "image",
                "max_resolution": "2048x2048",
                "supported_formats": ["png", "jpeg"],
                "additional_constraints": {
                    "styles": [
                        "photographic", "artistic", "digital-art",
                        "cinematic", "comic-book", "fantasy-art",
                        "watercolor", "oil-painting"
                    ],
                    "quality_presets": ["standard", "premium"],
                    "max_images_per_request": 4,
                    "seed_support": True,
                    "negative_prompt": True
                }
            },
            "firefly-v3": {
                "type": "image",
                "max_resolution": "4096x4096",
                "supported_formats": ["png", "jpeg", "webp"],
                "additional_constraints": {
                    "styles": [
                        "photographic", "artistic", "digital-art",
                        "cinematic", "comic-book", "fantasy-art",
                        "watercolor", "oil-painting", "3d-render",
                        "pixel-art", "isometric", "anime"
                    ],
                    "quality_presets": ["standard", "premium", "max"],
                    "max_images_per_request": 8,
                    "seed_support": True,
                    "negative_prompt": True,
                    "style_prompt": True,
                    "control_net": True
                }
            }
        }
        
        if not model_name:
            return MODEL_LIMITS
        return MODEL_LIMITS.get(model_name, {})
