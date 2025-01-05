from typing import Optional, Dict, Any
from huggingface_hub import HfApi
from . import BaseProvider

class HuggingFaceProvider(BaseProvider):
    """Provider class for HuggingFace models."""
    
    def __init__(self, api_token: Optional[str] = None):
        self.api_token = api_token
        self.api = HfApi(token=api_token)
        
    def get_model_limits(self, model_name: str) -> Dict[str, Any]:
        """Get the token limits for a specific HuggingFace model."""
        # Default limits for HuggingFace models
        return {
            "model_max_tokens": 2048,  # Default max tokens
            "is_within_limit": True,
            "total_tokens": 0,
            "recommended_batch_size": 2048,
            "batches": [],
            "model_supported_formats": ["text"],
            "model_additional_constraints": {}
        }
    
    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """List available HuggingFace models and their limits."""
        try:
            # Filter for text generation models
            models = self.api.list_models(filter="text-generation")
            
            model_limits = {}
            for model in models:
                limits = self.get_model_limits(model.modelId)
                if limits:
                    model_limits[model.modelId] = limits
            
            return model_limits
        except Exception as e:
            return {}
