"""Stability AI provider integration."""

from typing import Dict, Any, Optional
import stability_sdk
from .provider_template import ProviderTemplate

class StabilityProvider(ProviderTemplate):
    """Stability AI provider for image and video generation."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Stability AI provider with API key."""
        self.api_key = api_key
        if api_key:
            self.client = stability_sdk.client.StabilityInference(
                key=api_key,
                verbose=True
            )
    
    def get_model_limits(self, model_name: str) -> Dict[str, Any]:
        """Get the limits for Stability AI models."""
        MODEL_LIMITS = {
            # Image Generation Models
            "stable-diffusion-xl-1024-v1-0": {
                "type": "image",
                "max_resolution": "1024x1024",
                "supported_formats": ["png", "jpeg"],
                "additional_constraints": {
                    "prompt_length": 2000,
                    "negative_prompt": True,
                    "guidance_scale_range": [1.0, 20.0],
                    "steps_range": [10, 150],
                    "seed": True,
                    "styles": ["photographic", "digital-art", "anime"],
                    "safety_checker": True
                }
            },
            "stable-diffusion-xl-v1-0": {
                "type": "image",
                "max_resolution": "1024x1024",
                "supported_formats": ["png", "jpeg"],
                "additional_constraints": {
                    "prompt_length": 2000,
                    "negative_prompt": True,
                    "guidance_scale_range": [1.0, 20.0],
                    "steps_range": [10, 150],
                    "seed": True,
                    "styles": ["photographic", "digital-art", "anime"],
                    "safety_checker": True
                }
            },
            "stable-diffusion-512-v2-1": {
                "type": "image",
                "max_resolution": "512x512",
                "supported_formats": ["png", "jpeg"],
                "additional_constraints": {
                    "prompt_length": 2000,
                    "negative_prompt": True,
                    "guidance_scale_range": [1.0, 20.0],
                    "steps_range": [10, 150],
                    "seed": True,
                    "styles": ["photographic", "digital-art", "anime"],
                    "safety_checker": True
                }
            },
            
            # Image-to-Image Models
            "stable-diffusion-image-to-image": {
                "type": "image",
                "max_resolution": "1024x1024",
                "supported_formats": ["png", "jpeg"],
                "additional_constraints": {
                    "prompt_length": 2000,
                    "negative_prompt": True,
                    "guidance_scale_range": [1.0, 20.0],
                    "steps_range": [10, 150],
                    "seed": True,
                    "image_strength_range": [0.0, 1.0],
                    "styles": ["photographic", "digital-art", "anime"],
                    "safety_checker": True
                }
            },
            
            # Inpainting Models
            "stable-diffusion-inpainting": {
                "type": "image",
                "max_resolution": "1024x1024",
                "supported_formats": ["png", "jpeg"],
                "additional_constraints": {
                    "prompt_length": 2000,
                    "negative_prompt": True,
                    "guidance_scale_range": [1.0, 20.0],
                    "steps_range": [10, 150],
                    "seed": True,
                    "mask_required": True,
                    "styles": ["photographic", "digital-art", "anime"],
                    "safety_checker": True
                }
            },
            
            # Video Models
            "stable-video-diffusion": {
                "type": "video",
                "max_resolution": "1024x576",
                "supported_formats": ["mp4"],
                "additional_constraints": {
                    "prompt_length": 2000,
                    "negative_prompt": True,
                    "guidance_scale_range": [1.0, 20.0],
                    "steps_range": [10, 50],
                    "fps": 24,
                    "duration_range": [2, 16],  # seconds
                    "motion_bucket_id_range": [1, 255],
                    "seed": True
                }
            }
        }
        
        if not model_name:
            return MODEL_LIMITS
        return MODEL_LIMITS.get(model_name, {})
