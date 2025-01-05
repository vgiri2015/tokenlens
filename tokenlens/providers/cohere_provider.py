"""Cohere API provider integration."""

from typing import Dict, Any, Optional
import cohere
from .provider_template import ProviderTemplate

class CohereProvider(ProviderTemplate):
    """Cohere provider for text generation, embeddings, and reranking."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Cohere provider with API key."""
        self.client = cohere.Client(api_key) if api_key else None
    
    def get_model_limits(self, model_name: str) -> Dict[str, Any]:
        """Get the limits for Cohere models."""
        MODEL_LIMITS = {
            # Text Generation Models
            "command": {
                "type": "text",
                "token_limit": 4096,
                "max_output_tokens": 2048,
                "additional_constraints": {
                    "temperature_range": [0.0, 5.0],
                    "top_p_range": [0.0, 1.0],
                    "top_k_range": [0, 500],
                    "frequency_penalty_range": [0.0, 1.0],
                    "presence_penalty_range": [0.0, 1.0],
                    "system_prompt": True,
                    "stream": True,
                    "json_mode": True
                }
            },
            "command-light": {
                "type": "text",
                "token_limit": 4096,
                "max_output_tokens": 2048,
                "additional_constraints": {
                    "temperature_range": [0.0, 5.0],
                    "top_p_range": [0.0, 1.0],
                    "top_k_range": [0, 500],
                    "frequency_penalty_range": [0.0, 1.0],
                    "presence_penalty_range": [0.0, 1.0],
                    "system_prompt": True,
                    "stream": True,
                    "json_mode": True
                }
            },
            "command-nightly": {
                "type": "text",
                "token_limit": 4096,
                "max_output_tokens": 2048,
                "additional_constraints": {
                    "temperature_range": [0.0, 5.0],
                    "top_p_range": [0.0, 1.0],
                    "top_k_range": [0, 500],
                    "frequency_penalty_range": [0.0, 1.0],
                    "presence_penalty_range": [0.0, 1.0],
                    "system_prompt": True,
                    "stream": True,
                    "json_mode": True
                }
            },
            
            # Embedding Models
            "embed-english-v3.0": {
                "type": "embedding",
                "token_limit": 512,
                "dimensions": 1024,
                "additional_constraints": {
                    "truncate": True,
                    "model_type": "english",
                    "encoding": "utf-8"
                }
            },
            "embed-multilingual-v3.0": {
                "type": "embedding",
                "token_limit": 512,
                "dimensions": 1024,
                "additional_constraints": {
                    "truncate": True,
                    "model_type": "multilingual",
                    "supported_languages": 100,
                    "encoding": "utf-8"
                }
            },
            
            # Reranking Models
            "rerank-english-v3.0": {
                "type": "rerank",
                "token_limit": 512,
                "additional_constraints": {
                    "max_chunks": 100,
                    "model_type": "english",
                    "top_n": 100,
                    "return_documents": True
                }
            },
            "rerank-multilingual-v3.0": {
                "type": "rerank",
                "token_limit": 512,
                "additional_constraints": {
                    "max_chunks": 100,
                    "model_type": "multilingual",
                    "supported_languages": 100,
                    "top_n": 100,
                    "return_documents": True
                }
            }
        }
        
        if not model_name:
            return MODEL_LIMITS
        return MODEL_LIMITS.get(model_name, {})
        
    def _count_tokens(self, content: str) -> int:
        """Count tokens using Cohere's tokenizer."""
        if self.client:
            try:
                return self.client.tokenize(text=content).length
            except:
                pass
        return super()._count_tokens(content)  # Fallback to word-based counting
