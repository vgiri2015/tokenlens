from typing import Dict, Any
from .provider_template import ProviderTemplate

class RealmProvider(ProviderTemplate):
    def __init__(self):
        super().__init__()
        self.provider_name = "realm"
        
    def check_image_limits(self, request: Dict[str, Any], model: str) -> Dict[str, Any]:
        """Check image generation limits for Realm AI."""
        config = self._get_model_config(model)
        prompt = request.get("prompt", "")
        max_resolution = config.get("max_resolution", "1920x1080")
        
        return {
            "prompt_length": len(prompt),
            "max_resolution": max_resolution,
            "provider": self.provider_name,
            "model": model
        }

class StarrytarsProvider(ProviderTemplate):
    def __init__(self):
        super().__init__()
        self.provider_name = "starrytars"
        
    def check_image_limits(self, request: Dict[str, Any], model: str) -> Dict[str, Any]:
        """Check image generation limits for Starrytars."""
        config = self._get_model_config(model)
        prompt = request.get("prompt", "")
        max_resolution = config.get("max_resolution", "1920x1080")
        
        return {
            "prompt_length": len(prompt),
            "max_resolution": max_resolution,
            "provider": self.provider_name,
            "model": model
        }
