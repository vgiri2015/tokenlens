# TokenLens

A lightweight Python library for accurate token counting and limit validation across various LLM providers. TokenLens helps developers avoid costly API failures by providing accurate token counting before making API calls.

## Features

- **Modular Design**: Provider dependencies are only imported when needed
- **Accurate Token Counting**: Provider-specific tokenizer implementations
- **Proactive Limit Checking**: Validate content length before API calls
- **Smart Token Management**: Encoding and decoding support
- **Provider Independence**: Each provider can be installed separately

## Installation

Install the base package:
```bash
pip install tokenlens
```

Install with specific provider dependencies:
```bash
# For OpenAI support
pip install tokenlens[openai]

# For Anthropic support
pip install tokenlens[anthropic]

# For all providers
pip install tokenlens[all]
```

## Quick Start

### Basic Token Counting with OpenAI

```python
from tokenlens import OpenAITokenizer
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()

# Initialize tokenizer
tokenizer = OpenAITokenizer(model_name="gpt-4")

# Count tokens
text = "Hello, world!"
token_count = tokenizer.count_tokens(text)
print(f"Token count: {token_count}")

# Encode text to tokens
tokens = tokenizer.encode(text)
print(f"Tokens: {tokens}")

# Decode tokens back to text
decoded_text = tokenizer.decode(tokens)
print(f"Decoded text: {decoded_text}")
```

### Using Other Providers

TokenLens uses a modular design where provider dependencies are only imported when needed:

```python
# For Anthropic
from tokenlens.tokenizers import AnthropicTokenizer
tokenizer = AnthropicTokenizer()

# For Mistral
from tokenlens.tokenizers import MistralTokenizer
tokenizer = MistralTokenizer()
```

## Provider-Specific Dependencies

TokenLens uses a modular dependency system. Each provider's dependencies are optional and must be installed separately:

```bash
# For Mistral support
pip install tokenlens[mistral]

# For OpenAI support
pip install tokenlens[openai]

# For Anthropic support
pip install tokenlens[anthropic]

# For all providers
pip install tokenlens[all]
```

If you try to use a provider without installing its dependencies, TokenLens will:
1. Attempt to use a fallback implementation if available
2. Raise an informative error message if the provider's functionality is required

Example with Mistral:
```python
from tokenlens import MistralTokenizer

# This requires the mistralai package
tokenizer = MistralTokenizer(api_key="your-api-key")

# Count tokens (will use fallback if mistralai not installed)
text = "Hello, world!"
token_count = tokenizer.count_tokens(text)

# Get model limits
model_limits = tokenizer.get_model_limits("mistral-medium")
print(f"Model limits: {model_limits}")
```

## The Problem TokenLens Solves

Traditional AI application development often faces these challenges:

1. **Reactive Error Handling**: Without TokenLens
   ```python
   try:
       response = openai.ChatCompletion.create(
           model="gpt-4",
           messages=[{"role": "user", "content": very_long_text}]
       )
   except openai.error.InvalidRequestError as e:
       # Now you have to handle the API failure after it occurs
       pass
   ```

2. **Manual Content Splitting**: Without proper token counting
   ```python
   # Naive character-based splitting that might break context
   chunk_size = 1000
   chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
   ```

## The TokenLens Solution

TokenLens provides accurate token counting and limit validation:

```python
from tokenlens import OpenAITokenizer

tokenizer = OpenAITokenizer(model_name="gpt-4")

# Check token count before making API calls
token_count = tokenizer.count_tokens(very_long_text)

if token_count <= tokenizer.model_max_tokens:
    # Safe to make API call
    response = openai.ChatCompletion.create(...)
else:
    # Handle content that exceeds token limit
    print(f"Content too long: {token_count} tokens")
```

## Token Limit Validation

TokenLens makes it easy to check token counts against model limits:

```python
from tokenlens import OpenAITokenizer

# Initialize tokenizer
tokenizer = OpenAITokenizer(model_name="gpt-4")

# Get model limits
model_limits = tokenizer.get_model_limits()
print(f"Model limits: {model_limits}")
# Output: {'token_limit': 8192, 'max_response_tokens': 4096}

# Check if text is within limits
text = "Your long text here..."
token_count = tokenizer.count_tokens(text)

if token_count <= model_limits['token_limit']:
    print(f"Text is within limits ({token_count} tokens)")
else:
    print(f"Text exceeds model limit: {token_count} > {model_limits['token_limit']}")

# For chat completions, account for response tokens
available_tokens = model_limits['token_limit'] - model_limits['max_response_tokens']
if token_count <= available_tokens:
    print("Safe to send to chat completion API")
else:
    print("Need to reduce input tokens to leave room for response")
```

This proactive validation helps you:
- Avoid costly API failures due to token limits
- Ensure enough tokens are reserved for model responses
- Make informed decisions about content splitting

## Why Use TokenLens?

1. **Accurate Token Counting**: Get exact token counts before making API calls
2. **Provider-Specific Tokenization**: Each provider uses its own tokenization rules
3. **Modular Dependencies**: Only install the provider packages you need
4. **Simple Interface**: Consistent API across all providers
5. **Proactive Validation**: Check limits before making expensive API calls

## Development

1. Clone the repository:
```bash
git clone https://github.com/yourusername/tokenlens.git
cd tokenlens
```

2. Install development dependencies:
```bash
pip install -e ".[dev]"
```

3. Run tests:
```bash
# Run all tests
pytest

# Run specific provider tests
python tests/test_openai_limits.py
```

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
