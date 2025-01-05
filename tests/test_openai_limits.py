#!/usr/bin/env python3
"""Test token limits for OpenAI models.

This test requires the OpenAI dependencies:
    pip install tokenlens[openai]
"""

import os
import logging
import sys
from importlib.util import find_spec
from dotenv import load_dotenv
from tokenlens.tokenizers.openai_tokenizer import OpenAITokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

def check_openai_dependencies():
    """Check if OpenAI dependencies are installed."""
    if find_spec('openai') is None:
        logger.error(
            "OpenAI package not found. Install OpenAI dependencies with:\n"
            "    pip install tokenlens[openai]"
        )
        sys.exit(1)

def generate_test_text(length: int) -> str:
    """Generate test text of approximately given length."""
    base_text = "This is a test sentence for token counting. "
    return base_text * (length // len(base_text) + 1)

def test_openai_token_limits():
    """Test OpenAI token limits."""
    # Check dependencies first
    check_openai_dependencies()
    
    logger.info("Starting OpenAI token limits test")
    
    # Model configurations from OpenAI's documentation
    models_config = {
        "gpt-4": {
            "token_limit": 8192,
            "max_response_tokens": 4096
        },
        "gpt-4-32k": {
            "token_limit": 32768,
            "max_response_tokens": 8192
        },
        "gpt-3.5-turbo": {
            "token_limit": 4096,
            "max_response_tokens": 2048
        }
    }
    
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error(
            "No OpenAI API key found. Set the OPENAI_API_KEY environment variable:\n"
            "    export OPENAI_API_KEY=your-key-here"
        )
        sys.exit(1)
    
    logger.info(f"Testing {len(models_config)} OpenAI models")
    
    for model_name, limits in models_config.items():
        logger.info(f"\nTesting {model_name}")
        logger.info(f"Model limits:")
        logger.info(f"  Max tokens: {limits['token_limit']}")
        logger.info(f"  Max response tokens: {limits['max_response_tokens']}")
        
        try:
            # Initialize OpenAI tokenizer
            tokenizer = OpenAITokenizer(model_name=model_name, api_key=api_key)
            
            test_cases = [
                ("Small text", "Hello, world!"),
                ("Medium text", generate_test_text(1000)),
                ("Large text", generate_test_text(10000)),
                ("Max input text", generate_test_text(limits['token_limit'] * 4))
            ]
            
            for test_name, text in test_cases:
                try:
                    # Count tokens
                    token_count = tokenizer.count_tokens(text)
                    status = "PASS" if token_count <= limits['token_limit'] else "FAIL"
                    logger.info(f"{test_name}: {token_count} tokens [{status}]")
                    
                    # Test encoding/decoding
                    encoded = tokenizer.encode(text[:100])
                    decoded = tokenizer.decode(encoded)
                    encoding_status = "PASS" if decoded.strip() else "FAIL"
                    logger.info(f"{test_name} encoding/decoding: [{encoding_status}]")
                    
                except Exception as e:
                    logger.error(f"Error in {test_name}: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error initializing tokenizer for {model_name}: {str(e)}")

if __name__ == "__main__":
    test_openai_token_limits()
