"""TokenLens: A library for accurate token counting and limit validation across various LLM providers."""

from .tokenizers import BaseTokenizer, OpenAITokenizer

# Optional tokenizers
try:
    from .tokenizers import AnthropicTokenizer
except ImportError:
    AnthropicTokenizer = None

try:
    from .tokenizers import MistralTokenizer
except ImportError:
    MistralTokenizer = None

try:
    from .tokenizers import CohereTokenizer
except ImportError:
    CohereTokenizer = None

try:
    from .tokenizers import MetaTokenizer
except ImportError:
    MetaTokenizer = None

try:
    from .tokenizers import GoogleTokenizer
except ImportError:
    GoogleTokenizer = None

try:
    from .tokenizers import HuggingFaceTokenizer
except ImportError:
    HuggingFaceTokenizer = None

try:
    from .tokenizers import AI21Tokenizer
except ImportError:
    AI21Tokenizer = None

try:
    from .tokenizers import DeepMindTokenizer
except ImportError:
    DeepMindTokenizer = None

try:
    from .tokenizers import QwenTokenizer
except ImportError:
    QwenTokenizer = None

try:
    from .tokenizers import StanfordTokenizer
except ImportError:
    StanfordTokenizer = None

__version__ = "0.1.6"

__all__ = [
    "BaseTokenizer",
    "OpenAITokenizer",
    "AnthropicTokenizer",
    "MistralTokenizer",
    "CohereTokenizer",
    "MetaTokenizer",
    "GoogleTokenizer",
    "HuggingFaceTokenizer",
    "AI21Tokenizer",
    "DeepMindTokenizer",
    "QwenTokenizer",
    "StanfordTokenizer",
]
