from typing import Dict, Any, Optional
import tiktoken
from . import BaseProvider
from .provider_template import ProviderTemplate
import os

class QwenProvider(ProviderTemplate):
    """Provider implementation for Qwen models."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Qwen provider."""
        super().__init__(api_key or os.getenv("QWEN_API_KEY"))
        # Initialize Qwen-specific tokenizer
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # Using OpenAI's tokenizer as example
    
    def check_text_limits(
        self, 
        content: str, 
        model: str,
        additional_constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Check text model limits for Qwen models."""
        # Get model configuration
        model_config = self.config_loader.get_model_details("text", "qwen", model)
        
        # Calculate tokens using Qwen's tokenizer
        total_tokens = len(self.tokenizer.encode(content))
        
        # Check if within limits
        is_within_limit = total_tokens <= model_config["token_limit"]
        
        # Calculate batches if needed
        batches = None
        if not is_within_limit:
            batch_size = model_config["token_limit"] // 2
            if additional_constraints and "batch_size" in additional_constraints:
                batch_size = additional_constraints["batch_size"]
            
            # Calculate character positions for batches
            batches = []
            tokens = self.tokenizer.encode(content)
            current_batch = 0
            current_tokens = 0
            start_char = 0
            
            for i, token in enumerate(tokens):
                current_tokens += 1
                if current_tokens >= batch_size or i == len(tokens) - 1:
                    # Decode tokens to find character position
                    end_char = len(self.tokenizer.decode(tokens[:i+1]))
                    batches.append({
                        "batch": current_batch + 1,
                        "tokens": current_tokens,
                        "start_char": start_char,
                        "end_char": end_char
                    })
                    start_char = end_char
                    current_tokens = 0
                    current_batch += 1
        
        return {
            "total_tokens": total_tokens,
            "is_within_limit": is_within_limit,
            "model_max_tokens": model_config["token_limit"],
            "batches": batches,
            "model_additional_constraints": model_config.get("additional_constraints", {})
        }
