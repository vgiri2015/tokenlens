# TokenLens API Documentation

TokenLens provides a simple, consistent API for token counting and validation across various LLM providers. This document outlines the available APIs and their usage.

## Core Interface

All tokenizers in TokenLens implement the `BaseTokenizer` interface:

```python
from tokenlens import BaseTokenizer

class BaseTokenizer:
    def encode(self, text: str) -> List[int]:
        """Encode text into tokens.
        
        Args:
            text: The input text to encode
            
        Returns:
            A list of integer token IDs
        """
        pass
    
    def decode(self, tokens: List[int]) -> str:
        """Decode tokens back into text.
        
        Args:
            tokens: List of integer token IDs
            
        Returns:
            The decoded text
        """
        pass
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the text.
        
        Args:
            text: The input text to count tokens for
            
        Returns:
            The number of tokens in the text
        """
        return len(self.encode(text))
```

## Provider-Specific Tokenizers

TokenLens provides tokenizer implementations for various LLM providers. Each tokenizer is loaded only when imported, ensuring minimal dependencies.

### OpenAI Tokenizer

```python
from tokenlens import OpenAITokenizer

tokenizer = OpenAITokenizer(model_name="gpt-4")
text = "Hello, world!"

# Count tokens
token_count = tokenizer.count_tokens(text)  # Returns: 4

# Encode text to tokens
tokens = tokenizer.encode(text)  # Returns: [9906, 11, 4435, 0]

# Decode tokens back to text
decoded = tokenizer.decode(tokens)  # Returns: "Hello, world!"
```

### Anthropic Tokenizer

```python
from tokenlens import AnthropicTokenizer

tokenizer = AnthropicTokenizer()
text = "Hello, world!"

tokens = tokenizer.encode(text)
token_count = tokenizer.count_tokens(text)
decoded = tokenizer.decode(tokens)
```

### Mistral Tokenizer

```python
from tokenlens import MistralTokenizer

tokenizer = MistralTokenizer()
text = "Hello, world!"

tokens = tokenizer.encode(text)
token_count = tokenizer.count_tokens(text)
decoded = tokenizer.decode(tokens)
```

### Other Available Tokenizers

TokenLens provides tokenizers for many popular LLM providers:

```python
from tokenlens import (
    CohereTokenizer,
    MetaTokenizer,          # For Meta's LLaMA models
    GoogleTokenizer,        # For Google's Gemini models
    HuggingFaceTokenizer,   # For Hugging Face models
    AI21Tokenizer,         # For AI21's Jurassic models
    DeepMindTokenizer,     # For DeepMind's models
    QwenTokenizer,         # For Qwen models
    StanfordTokenizer,     # For Stanford's models
)
```

## Installation

TokenLens uses a modular dependency system. Install only the provider packages you need:

```bash
# Base package
pip install tokenlens

# Provider-specific installations
pip install tokenlens[openai]      # For OpenAI support
pip install tokenlens[anthropic]   # For Anthropic support
pip install tokenlens[mistral]     # For Mistral support
pip install tokenlens[cohere]      # For Cohere support
pip install tokenlens[meta]        # For Meta/LLaMA support
pip install tokenlens[google]      # For Google/Gemini support
pip install tokenlens[all]         # For all providers
```

## Error Handling

TokenLens uses a hierarchical error handling system:

```python
from tokenlens import OpenAITokenizer

try:
    tokenizer = OpenAITokenizer(model_name="gpt-4")
    token_count = tokenizer.count_tokens("Hello, world!")
except ValueError as e:
    # Handle initialization or token counting errors
    print(f"Error: {str(e)}")
```

Common exceptions:
- `ValueError`: Invalid input or configuration
- `ImportError`: Provider package not installed
- `RuntimeError`: Tokenizer initialization failed

## Best Practices

1. **Lazy Loading**: Import only the tokenizers you need to minimize dependencies
   ```python
   # Good - only loads OpenAI dependencies
   from tokenlens import OpenAITokenizer
   
   # Bad - loads all provider dependencies
   from tokenlens import *
   ```

2. **Token Validation**: Check token counts before making API calls
   ```python
   tokenizer = OpenAITokenizer(model_name="gpt-4")
   text = "Very long text..."
   
   token_count = tokenizer.count_tokens(text)
   if token_count <= tokenizer.model_max_tokens:
       # Safe to make API call
       pass
   ```

3. **Environment Variables**: Use environment variables for API keys
   ```python
   from dotenv import load_dotenv
   load_dotenv()  # Load API keys from .env file
   
   tokenizer = OpenAITokenizer(model_name="gpt-4")
   ```

## Contributing

To add support for a new provider:

1. Create a new tokenizer class that implements `BaseTokenizer`
2. Add provider-specific dependencies to `pyproject.toml`
3. Update the tokenizer imports in `__init__.py`
4. Add tests for the new tokenizer
5. Update the documentation

See [CONTRIBUTING.md](../CONTRIBUTING.md) for more details.
