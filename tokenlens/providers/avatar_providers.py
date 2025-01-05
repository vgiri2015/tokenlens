from typing import Dict, Any
from .provider_template import ProviderTemplate

class SynthesiaProvider(ProviderTemplate):
    def __init__(self):
        super().__init__()
        self.provider_name = "synthesia"
        
    def check_avatar_limits(self, request: Dict[str, Any], model: str) -> Dict[str, Any]:
        """Check avatar generation limits for Synthesia."""
        config = self._get_model_config(model)
        script = request.get("script", "")
        duration = request.get("duration", 0)
        
        max_duration = int(config.get("max_duration", 300))
        max_resolution = config.get("max_resolution", "1920x1080")
        
        return {
            "script_length": len(script),
            "duration": duration,
            "max_duration": max_duration,
            "within_duration_limit": duration <= max_duration,
            "max_resolution": max_resolution,
            "provider": self.provider_name,
            "model": model
        }

class DIDProvider(ProviderTemplate):
    def __init__(self):
        super().__init__()
        self.provider_name = "d-id"
        
    def check_avatar_limits(self, request: Dict[str, Any], model: str) -> Dict[str, Any]:
        """Check avatar generation limits for D-ID."""
        config = self._get_model_config(model)
        script = request.get("script", "")
        duration = request.get("duration", 0)
        
        max_duration = int(config.get("max_duration", 300))
        max_resolution = config.get("max_resolution", "1920x1080")
        
        return {
            "script_length": len(script),
            "duration": duration,
            "max_duration": max_duration,
            "within_duration_limit": duration <= max_duration,
            "max_resolution": max_resolution,
            "provider": self.provider_name,
            "model": model
        }

class ReplikaProvider(ProviderTemplate):
    def __init__(self):
        super().__init__()
        self.provider_name = "replika"
        
    def check_avatar_limits(self, request: Dict[str, Any], model: str) -> Dict[str, Any]:
        """Check avatar generation limits for Replika."""
        config = self._get_model_config(model)
        script = request.get("script", "")
        duration = request.get("duration", 0)
        
        max_duration = int(config.get("max_duration", 300))
        max_resolution = config.get("max_resolution", "1280x720")
        
        return {
            "script_length": len(script),
            "duration": duration,
            "max_duration": max_duration,
            "within_duration_limit": duration <= max_duration,
            "max_resolution": max_resolution,
            "provider": self.provider_name,
            "model": model
        }
