from typing import Dict, Any
from .provider_template import ProviderTemplate

class MicrosoftProvider(ProviderTemplate):
    def __init__(self):
        super().__init__()
        self.provider_name = "microsoft"
        
    def check_image_limits(self, request: Dict[str, Any], model: str) -> Dict[str, Any]:
        """Check image generation limits for Microsoft Designer."""
        config = self._get_model_config(model)
        prompt = request.get("prompt", "")
        max_resolution = config.get("max_resolution", "2048x2048")
        supported_formats = config.get("supported_formats", ["png", "jpeg", "psd"])
        
        return {
            "prompt_length": len(prompt),
            "max_resolution": max_resolution,
            "supported_formats": supported_formats,
            "provider": self.provider_name,
            "model": model
        }
        
    def check_video_limits(self, request: Dict[str, Any], model: str) -> Dict[str, Any]:
        """Check video generation limits for Microsoft Direct2V."""
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
        
    def list_models(self) -> Dict[str, Any]:
        """List available Microsoft models."""
        return {
            "image_models": ["designer"],
            "video_models": ["direct2v"],
            "provider": self.provider_name
        }
