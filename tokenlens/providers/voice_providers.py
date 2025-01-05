from typing import Dict, Any
from .provider_template import ProviderTemplate

class FugattoProvider(ProviderTemplate):
    def __init__(self):
        super().__init__()
        self.provider_name = "fugatto"
        
    def check_voice_limits(self, text: str, model: str) -> Dict[str, Any]:
        """Check voice generation limits for Fugatto."""
        config = self._get_model_config(model)
        max_duration = int(config.get("max_duration", 300))
        supported_formats = config.get("supported_formats", ["mp3", "wav"])
        
        # Approximate duration based on character count (rough estimate)
        estimated_duration = len(text) / 20  # ~20 chars per second
        
        return {
            "text_length": len(text),
            "estimated_duration": estimated_duration,
            "max_duration": max_duration,
            "within_duration_limit": estimated_duration <= max_duration,
            "supported_formats": supported_formats,
            "provider": self.provider_name,
            "model": model
        }
        
    def list_models(self) -> Dict[str, Any]:
        """List available Fugatto models."""
        return {
            "voice_models": ["v1"],
            "provider": self.provider_name
        }
