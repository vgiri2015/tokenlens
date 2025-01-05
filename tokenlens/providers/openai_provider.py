from typing import Dict, Any, Optional
import openai
from . import BaseProvider

class OpenAIProvider(BaseProvider):
    """OpenAI API provider integration."""
    
    def __init__(self, api_key: Optional[str] = None):
        if api_key:
            openai.api_key = api_key
    
    def get_model_limits(self, model_name: str) -> Dict[str, Any]:
        """Get the limits for a specific OpenAI model."""
        # OpenAI model limits
        MODEL_LIMITS = {
            # Text Models
            "gpt-4": {"token_limit": 8192, "type": "text"},
            "gpt-4-32k": {"token_limit": 32768, "type": "text"},
            "gpt-4-1106-preview": {"token_limit": 128000, "type": "text"},
            "gpt-4-vision-preview": {"token_limit": 128000, "type": "text"},
            "gpt-3.5-turbo": {"token_limit": 4096, "type": "text"},
            "gpt-3.5-turbo-16k": {"token_limit": 16384, "type": "text"},
            "gpt-3.5-turbo-1106": {"token_limit": 16384, "type": "text"},
            
            # Image Models
            "dall-e-3": {
                "type": "image",
                "max_resolution": "1024x1024",
                "supported_formats": ["png", "jpeg"],
                "additional_constraints": {
                    "quality": ["standard", "hd"],
                    "style": ["vivid", "natural"]
                }
            },
            "dall-e-2": {
                "type": "image",
                "max_resolution": "1024x1024",
                "supported_formats": ["png", "jpeg"],
                "additional_constraints": {
                    "size": ["256x256", "512x512", "1024x1024"]
                }
            },
            
            # Audio/Voice Models
            "whisper-1": {
                "type": "voice",
                "supported_formats": ["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"],
                "max_duration": "300", # 5 minutes
                "additional_constraints": {
                    "file_size": "25MB",
                    "languages": "multilingual"
                }
            },
            "tts-1": {
                "type": "voice",
                "supported_formats": ["mp3", "opus", "aac", "flac"],
                "additional_constraints": {
                    "voices": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
                    "max_text_length": 4096
                }
            },
            
            "tts-1-hd": {
                "type": "voice",
                "supported_formats": ["mp3", "opus", "aac", "flac"],
                "additional_constraints": {
                    "voices": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
                    "max_text_length": 4096,
                    "quality": "high-definition"
                }
            }
        }
        
        return MODEL_LIMITS.get(model_name, {})
    
    def check_text_limits(self, model_name: str, text: str) -> Dict[str, Any]:
        """Check if text is within the model's token limits."""
        model_limits = self.get_model_limits(model_name)
        if not model_limits or model_limits.get("type") != "text":
            return {"error": f"Model {model_name} not found or is not a text model"}
        
        # Use tiktoken to count tokens
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model(model_name)
            token_count = len(encoding.encode(text))
            
            token_limit = model_limits["token_limit"]
            is_within_limit = token_count <= token_limit
            
            return {
                "token_count": token_count,
                "token_limit": token_limit,
                "is_within_limit": is_within_limit,
                "model": model_name
            }
        except Exception as e:
            return {"error": f"Error counting tokens: {str(e)}"}
    
    def check_image_limits(self, model_name: str, image_info: Dict[str, Any]) -> Dict[str, Any]:
        """Check if image generation request is within the model's limits."""
        model_limits = self.get_model_limits(model_name)
        if not model_limits or model_limits.get("type") != "image":
            return {"error": f"Model {model_name} not found or is not an image model"}
        
        # Extract image info
        size = image_info.get("size", "1024x1024")
        format = image_info.get("format", "png").lower()
        quality = image_info.get("quality", "standard")
        style = image_info.get("style", "vivid")
        
        # Check constraints
        constraints = model_limits.get("additional_constraints", {})
        errors = []
        
        if "size" in constraints and size not in constraints["size"]:
            errors.append(f"Invalid size. Supported sizes: {constraints['size']}")
        
        if format not in model_limits["supported_formats"]:
            errors.append(f"Invalid format. Supported formats: {model_limits['supported_formats']}")
        
        if "quality" in constraints and quality not in constraints["quality"]:
            errors.append(f"Invalid quality. Supported qualities: {constraints['quality']}")
        
        if "style" in constraints and style not in constraints["style"]:
            errors.append(f"Invalid style. Supported styles: {constraints['style']}")
        
        if errors:
            return {"error": "; ".join(errors)}
        
        return {
            "is_within_limit": True,
            "model": model_name,
            "supported_formats": model_limits["supported_formats"],
            "max_resolution": model_limits["max_resolution"],
            "additional_constraints": constraints
        }
    
    def check_voice_limits(self, model_name: str, audio_info: Dict[str, Any]) -> Dict[str, Any]:
        """Check if audio processing request is within the model's limits."""
        model_limits = self.get_model_limits(model_name)
        if not model_limits or model_limits.get("type") != "voice":
            return {"error": f"Model {model_name} not found or is not a voice model"}
        
        # Extract audio info
        format = audio_info.get("format", "mp3").lower()
        duration = float(audio_info.get("duration", 0))  # in seconds
        file_size = float(audio_info.get("file_size", 0))  # in MB
        
        # Check constraints
        errors = []
        
        if format not in model_limits["supported_formats"]:
            errors.append(f"Invalid format. Supported formats: {model_limits['supported_formats']}")
        
        max_duration = float(model_limits["max_duration"])
        if duration > max_duration:
            errors.append(f"Duration exceeds limit of {max_duration} seconds")
        
        max_file_size = float(model_limits["additional_constraints"]["file_size"].replace("MB", ""))
        if file_size > max_file_size:
            errors.append(f"File size exceeds limit of {max_file_size}MB")
        
        if errors:
            return {"error": "; ".join(errors)}
        
        return {
            "is_within_limit": True,
            "model": model_name,
            "supported_formats": model_limits["supported_formats"],
            "max_duration": model_limits["max_duration"],
            "max_file_size": model_limits["additional_constraints"]["file_size"]
        }
    
    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """List all available OpenAI models and their limits."""
        try:
            models = openai.models.list()
            model_limits = {}
            
            for model in models.data:
                limits = self.get_model_limits(model.id)
                if limits:
                    model_limits[model.id] = limits
            
            return model_limits
        except Exception as e:
            return {}
