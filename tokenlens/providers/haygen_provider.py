from typing import Dict, Any, Optional
from .provider_template import ProviderTemplate

class HaygenProvider(ProviderTemplate):
    def __init__(self):
        super().__init__()
        # Model-specific limits
        self.limits = {
            "haygen-avatar-v1": {
                "max_script_chars": 2000,
                "max_duration": 300,  # seconds
                "supported_resolutions": ["720p", "1080p", "4k"]
            },
            "haygen-video-v1": {
                "max_script_chars": 5000,
                "max_duration": 300,  # seconds
                "max_resolution": "4k"
            },
            "haygen-voice-v1": {
                "max_script_chars": 3000,
                "max_duration": 300  # seconds
            }
        }

    def check_text_limits(self, text: str, model: str) -> Dict[str, Any]:
        char_count = len(text)
        model_limits = self.limits.get(model, {})
        max_chars = model_limits.get("max_script_chars", 2000)  # default

        return {
            "is_within_limit": char_count <= max_chars,
            "char_count": char_count,
            "max_chars": max_chars,
            "token_count": char_count // 4,  # Approximate for Haygen
            "max_tokens": max_chars // 4
        }

    def check_avatar_limits(self, content: Dict[str, Any], model: str) -> Dict[str, Any]:
        script = content.get("script", "")
        duration = content.get("duration", 0)
        resolution = content.get("resolution", "1080p")
        
        model_limits = self.limits.get(model, {})
        text_limits = self.check_text_limits(script, model)
        
        is_within_limit = (
            text_limits["is_within_limit"] and
            duration <= model_limits.get("max_duration", 300) and
            resolution in model_limits.get("supported_resolutions", ["1080p"])
        )

        return {
            "is_within_limit": is_within_limit,
            "char_count": text_limits["char_count"],
            "max_chars": text_limits["max_chars"],
            "duration": duration,
            "max_duration": model_limits.get("max_duration", 300),
            "resolution": resolution,
            "supported_resolutions": model_limits.get("supported_resolutions", ["1080p"])
        }

    def check_video_limits(self, content: Dict[str, Any], model: str) -> Dict[str, Any]:
        if isinstance(content, str):
            script = content
            duration = 60  # default duration
            resolution = "1080p"  # default resolution
        else:
            script = content.get("script", "")
            duration = content.get("duration", 60)
            resolution = content.get("resolution", "1080p")
        
        model_limits = self.limits.get(model, {})
        text_limits = self.check_text_limits(script, model)
        
        is_within_limit = (
            text_limits["is_within_limit"] and
            duration <= model_limits.get("max_duration", 300) and
            resolution <= model_limits.get("max_resolution", "4k")
        )

        return {
            "is_within_limit": is_within_limit,
            "char_count": text_limits["char_count"],
            "max_chars": text_limits["max_chars"],
            "duration": duration,
            "max_duration": model_limits.get("max_duration", 300),
            "resolution": resolution,
            "max_resolution": model_limits.get("max_resolution", "4k")
        }

    def check_voice_limits(self, text: str, model: str) -> Dict[str, Any]:
        text_limits = self.check_text_limits(text, model)
        model_limits = self.limits.get(model, {})
        
        # Estimate duration based on character count (rough approximation)
        estimated_duration = len(text) / 15  # ~15 chars per second
        
        is_within_limit = (
            text_limits["is_within_limit"] and
            estimated_duration <= model_limits.get("max_duration", 300)
        )

        return {
            "is_within_limit": is_within_limit,
            "char_count": text_limits["char_count"],
            "max_chars": text_limits["max_chars"],
            "estimated_duration": estimated_duration,
            "max_duration": model_limits.get("max_duration", 300)
        }

    def get_model_limits(self, model: str) -> Dict[str, Any]:
        return self.limits.get(model, {})
