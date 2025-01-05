from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseProvider(ABC):
    """Base class for provider API integrations."""
    
    @abstractmethod
    def get_model_limits(self, model_name: str) -> Dict[str, Any]:
        """Get the limits for a specific model."""
        pass
    
    @abstractmethod
    def check_text_limits(self, model_name: str, text: str) -> Dict[str, Any]:
        """Check if text is within the model's token limits."""
        pass
    
    @abstractmethod
    def check_image_limits(self, model_name: str, image_info: Dict[str, Any]) -> Dict[str, Any]:
        """Check if image generation request is within the model's limits."""
        pass
    
    @abstractmethod
    def check_video_limits(self, model_name: str, video_info: Dict[str, Any]) -> Dict[str, Any]:
        """Check if video generation request is within the model's limits."""
        pass
    
    @abstractmethod
    def check_avatar_limits(self, model_name: str, avatar_info: Dict[str, Any]) -> Dict[str, Any]:
        """Check if avatar generation request is within the model's limits."""
        pass
    
    @abstractmethod
    def check_voice_limits(self, model_name: str, audio_info: Dict[str, Any]) -> Dict[str, Any]:
        """Check if audio processing request is within the model's limits."""
        pass
    
    @abstractmethod
    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """List all available models and their limits."""
        pass
    
    @abstractmethod
    def refresh_models(self) -> None:
        """Refresh the list of available models."""
        pass

class TextProvider(BaseProvider):
    """Base class for text generation providers."""
    def check_image_limits(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Text provider does not support image generation")
    
    def check_video_limits(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Text provider does not support video generation")
    
    def check_avatar_limits(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Text provider does not support avatar generation")
    
    def check_voice_limits(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Text provider does not support voice generation")

class ImageProvider(BaseProvider):
    """Base class for image generation providers."""
    def check_text_limits(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Image provider does not support text generation")
    
    def check_video_limits(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Image provider does not support video generation")
    
    def check_avatar_limits(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Image provider does not support avatar generation")
    
    def check_voice_limits(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Image provider does not support voice generation")

class VideoProvider(BaseProvider):
    """Base class for video generation providers."""
    def check_text_limits(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Video provider does not support text generation")
    
    def check_image_limits(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Video provider does not support image generation")
    
    def check_avatar_limits(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Video provider does not support avatar generation")
    
    def check_voice_limits(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Video provider does not support voice generation")

class VoiceProvider(BaseProvider):
    """Base class for voice generation providers."""
    def check_text_limits(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Voice provider does not support text generation")
    
    def check_image_limits(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Voice provider does not support image generation")
    
    def check_video_limits(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Voice provider does not support video generation")
    
    def check_avatar_limits(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Voice provider does not support avatar generation")

class AvatarProvider(BaseProvider):
    """Base class for avatar generation providers."""
    def check_text_limits(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Avatar provider does not support text generation")
    
    def check_image_limits(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Avatar provider does not support image generation")
    
    def check_video_limits(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Avatar provider does not support video generation")
    
    def check_voice_limits(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Avatar provider does not support voice generation")
