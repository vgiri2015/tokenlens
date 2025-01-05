from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tokenlens",
    version="0.1.0",
    author="TokenLens Team",
    author_email="team@tokenlens.ai",
    description="A comprehensive API service to check and manage token limits for various LLM models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tokenlens/tokenlens",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "fastapi>=0.68.0",
        "pydantic>=1.8.0",
        "tiktoken>=0.3.0",
        "pyyaml>=5.4.1",
        "aiohttp>=3.8.0",
        "python-dotenv>=0.19.0"
    ],
)
