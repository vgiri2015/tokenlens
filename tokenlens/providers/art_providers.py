from typing import Dict, Any
from .provider_template import ProviderTemplate

class IdeogramProvider(ProviderTemplate):
    def __init__(self):
        super().__init__()
        self.provider_name = "ideogram"
        
    def check_image_limits(self, request: Dict[str, Any], model: str) -> Dict[str, Any]:
        """Check image generation limits for Ideogram."""
        config = self._get_model_config(model)
        prompt = request.get("prompt", "")
        max_resolution = config.get("max_resolution", "1024x1024")
        
        return {
            "prompt_length": len(prompt),
            "max_resolution": max_resolution,
            "provider": self.provider_name,
            "model": model
        }

class RunwayProvider(ProviderTemplate):
    def __init__(self):
        super().__init__()
        self.provider_name = "runway"
        
    def check_image_limits(self, request: Dict[str, Any], model: str) -> Dict[str, Any]:
        """Check image generation limits for Runway."""
        config = self._get_model_config(model)
        prompt = request.get("prompt", "")
        max_resolution = config.get("max_resolution", "1024x1024")
        
        return {
            "prompt_length": len(prompt),
            "max_resolution": max_resolution,
            "provider": self.provider_name,
            "model": model
        }
        
    def check_video_limits(self, request: Dict[str, Any], model: str) -> Dict[str, Any]:
        """Check video generation limits for Runway."""
        config = self._get_model_config(model)
        duration = request.get("duration", 0)
        max_duration = int(config.get("max_duration", 30))
        max_resolution = config.get("max_resolution", "1920x1080")
        
        return {
            "duration": duration,
            "max_duration": max_duration,
            "within_duration_limit": duration <= max_duration,
            "max_resolution": max_resolution,
            "provider": self.provider_name,
            "model": model
        }

class NightcafeProvider(ProviderTemplate):
    def __init__(self):
        super().__init__()
        self.provider_name = "nightcafe"
        
    def check_image_limits(self, request: Dict[str, Any], model: str) -> Dict[str, Any]:
        """Check image generation limits for Nightcafe."""
        config = self._get_model_config(model)
        prompt = request.get("prompt", "")
        max_resolution = config.get("max_resolution", "1920x1080")
        
        return {
            "prompt_length": len(prompt),
            "max_resolution": max_resolution,
            "provider": self.provider_name,
            "model": model
        }

class OpenArtProvider(ProviderTemplate):
    def __init__(self):
        super().__init__()
        self.provider_name = "openart"
        
    def check_image_limits(self, request: Dict[str, Any], model: str) -> Dict[str, Any]:
        """Check image generation limits for OpenArt."""
        config = self._get_model_config(model)
        prompt = request.get("prompt", "")
        max_resolution = config.get("max_resolution", "2048x2048")
        
        return {
            "prompt_length": len(prompt),
            "max_resolution": max_resolution,
            "provider": self.provider_name,
            "model": model
        }
