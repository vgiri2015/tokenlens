from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union

class ProviderTemplate(ABC):
    """Template class for implementing new providers."""
    
    @abstractmethod
    def __init__(self, api_key: Optional[str] = None):
        """Initialize provider with optional API key."""
        pass

    @abstractmethod
    def get_model_limits(self, model_name: str) -> Dict[str, Any]:
        """Get the limits for a specific model."""
        pass

    def check_text_limits(self, model: str, content: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Check text content against model limits."""
        limits = self.get_model_limits(model)
        if not limits or limits.get('type') != 'text':
            raise ValueError(f"Model {model} does not support text content")
            
        token_count = self._count_tokens(content)
        token_limit = limits.get('token_limit', 0)
        
        return {
            "valid": token_count <= token_limit,
            "token_count": token_count,
            "token_limit": token_limit,
            "model": model,
            "provider": self.__class__.__name__.replace('Provider', '').lower()
        }

    def check_image_limits(self, model: str, content: Dict[str, Any]) -> Dict[str, Any]:
        """Check image generation parameters against model limits."""
        limits = self.get_model_limits(model)
        if not limits or limits.get('type') != 'image':
            raise ValueError(f"Model {model} does not support image generation")
            
        max_res = limits.get('max_resolution', '1024x1024')
        max_w, max_h = map(int, max_res.split('x'))
        
        width = content.get('width', 0)
        height = content.get('height', 0)
        
        return {
            "valid": width <= max_w and height <= max_h,
            "width_limit": max_w,
            "height_limit": max_h,
            "model": model,
            "provider": self.__class__.__name__.replace('Provider', '').lower()
        }

    def check_video_limits(self, model: str, content: Dict[str, Any]) -> Dict[str, Any]:
        """Check video generation parameters against model limits."""
        limits = self.get_model_limits(model)
        if not limits or limits.get('type') != 'video':
            raise ValueError(f"Model {model} does not support video generation")
            
        max_duration = int(limits.get('max_duration', '60'))  # seconds
        duration = int(content.get('duration', 0))
        
        return {
            "valid": duration <= max_duration,
            "duration_limit": max_duration,
            "model": model,
            "provider": self.__class__.__name__.replace('Provider', '').lower()
        }

    def check_avatar_limits(self, model: str, content: Dict[str, Any]) -> Dict[str, Any]:
        """Check avatar generation parameters against model limits."""
        limits = self.get_model_limits(model)
        if not limits or limits.get('type') != 'avatar':
            raise ValueError(f"Model {model} does not support avatar generation")
            
        max_duration = int(limits.get('max_duration', '300'))  # seconds
        max_script_chars = int(limits.get('max_script_chars', '2000'))
        
        duration = int(content.get('duration', 0))
        script_length = len(content.get('script', ''))
        
        return {
            "valid": duration <= max_duration and script_length <= max_script_chars,
            "duration_limit": max_duration,
            "script_char_limit": max_script_chars,
            "model": model,
            "provider": self.__class__.__name__.replace('Provider', '').lower()
        }

    def check_voice_limits(self, model: str, content: Dict[str, Any]) -> Dict[str, Any]:
        """Check voice generation parameters against model limits."""
        limits = self.get_model_limits(model)
        if not limits or limits.get('type') != 'voice':
            raise ValueError(f"Model {model} does not support voice generation")
            
        max_chars = int(limits.get('max_chars', '4096'))
        text_length = len(content.get('text', ''))
        
        return {
            "valid": text_length <= max_chars,
            "char_limit": max_chars,
            "model": model,
            "provider": self.__class__.__name__.replace('Provider', '').lower()
        }

    def _count_tokens(self, content: Union[str, Dict[str, Any]]) -> int:
        """Count tokens in content. Override in provider-specific implementations."""
        if isinstance(content, str):
            # Default to word-based counting if no specific tokenizer
            return len(content.split())
        elif isinstance(content, dict):
            # For structured content, count tokens in all text fields
            total = 0
            for value in content.values():
                if isinstance(value, str):
                    total += len(value.split())
            return total
        return 0

    def get_supported_features(self) -> List[str]:
        """Get list of supported features for this provider."""
        features = set()
        for model_info in self.get_model_limits("").values():
            if isinstance(model_info, dict) and 'type' in model_info:
                features.add(model_info['type'])
        return list(features)

    def get_supported_models(self, feature: Optional[str] = None) -> List[str]:
        """Get list of supported models, optionally filtered by feature."""
        models = []
        for model, info in self.get_model_limits("").items():
            if isinstance(info, dict):
                if not feature or info.get('type') == feature:
                    models.append(model)
        return models
