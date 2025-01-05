import asyncio
import os
from dotenv import load_dotenv
from tokenlens import ModelProcessor

async def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Initialize the ModelProcessor
    processor = ModelProcessor()
    
    # Your OpenAI API key should be set in environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")

    # Test text completion
    text_content = """
    This is a sample text that we want to analyze. 
    We'll check if it fits within the model's token limits 
    and get information about token usage.
    """
    
    try:
        # Check text limits for GPT-4
        print("\n=== Testing GPT-4 Limits ===")
        result = processor.check_limits(
            provider="openai",
            content=text_content,
            model="gpt-4"
        )
        print(f"Token count: {result.get('total_tokens', 0)}")
        print(f"Within limits: {result.get('is_within_limit', False)}")
        print(f"Model max tokens: {result.get('model_max_tokens', 0)}")
        
        # Test token counting
        print("\n=== Token Counting ===")
        token_count = processor.count_tokens("openai", text_content)
        print(f"Direct token count: {token_count}")
        
        # Get provider information
        print("\n=== OpenAI Provider Info ===")
        provider_info = processor.get_provider_info("openai")
        print("Supported features:", provider_info.get("supported_features", []))
        print("Has tokenizer:", provider_info.get("has_tokenizer", False))
        
        # Test image generation limits
        print("\n=== Testing DALL-E Limits ===")
        image_content = {
            "prompt": "A beautiful sunset over mountains",
            "size": "1024x1024",
            "quality": "standard",
            "n": 1
        }
        
        image_result = processor.check_limits(
            provider="openai",
            content=image_content,
            model="dall-e-3",
            feature="image"
        )
        print("Image generation within limits:", image_result.get("is_within_limit", False))
        print("Supported formats:", image_result.get("supported_formats", []))
        print("Max resolution:", image_result.get("max_resolution", ""))
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
