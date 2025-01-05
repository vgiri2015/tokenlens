"""TokenLens tokenizers module."""

from .base import BaseTokenizer
from .openai_tokenizer import OpenAITokenizer

__all__ = ["BaseTokenizer", "OpenAITokenizer"]

# Optional tokenizers - import only if dependencies are available
try:
    from .anthropic_tokenizer import AnthropicTokenizer
    __all__.append("AnthropicTokenizer")
except ImportError:
    AnthropicTokenizer = None

try:
    from .mistral_tokenizer import MistralTokenizer
    __all__.append("MistralTokenizer")
except ImportError:
    MistralTokenizer = None

try:
    from .cohere_tokenizer import CohereTokenizer
    __all__.append("CohereTokenizer")
except ImportError:
    CohereTokenizer = None

try:
    from .meta_tokenizer import MetaTokenizer
    __all__.append("MetaTokenizer")
except ImportError:
    MetaTokenizer = None

try:
    from .google_tokenizer import GoogleTokenizer
    __all__.append("GoogleTokenizer")
except ImportError:
    GoogleTokenizer = None

try:
    from .ai21_tokenizer import AI21Tokenizer
    __all__.append("AI21Tokenizer")
except ImportError:
    AI21Tokenizer = None

try:
    from .deepmind_tokenizer import DeepMindTokenizer
    __all__.append("DeepMindTokenizer")
except ImportError:
    DeepMindTokenizer = None

try:
    from .huggingface_tokenizer import HuggingFaceTokenizer
    __all__.append("HuggingFaceTokenizer")
except ImportError:
    HuggingFaceTokenizer = None

try:
    from .qwen_tokenizer import QwenTokenizer
    __all__.append("QwenTokenizer")
except ImportError:
    QwenTokenizer = None

try:
    from .stanford_tokenizer import StanfordTokenizer
    __all__.append("StanfordTokenizer")
except ImportError:
    StanfordTokenizer = None
