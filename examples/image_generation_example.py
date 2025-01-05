import requests
import json
from PIL import Image
import io

def check_image_generation_limits(prompt: str, model: str = "dall-e-3", provider: str = "openai", api_key: str = None):
    """Check limits for image generation."""
    
    url = "http://localhost:8000/check-limits"
    
    payload = {
        "content": prompt,
        "model": model,
        "provider": provider,
        "model_type": "image",
        "api_key": api_key
    }
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nImage Generation Limits for {model} ({provider}):")
        print(f"Maximum Resolution: {result['model_max_resolution']}")
        print(f"Supported Formats: {', '.join(result['model_supported_formats'])}")
        
        if result['model_additional_constraints']:
            print("\nAdditional Constraints:")
            for key, value in result['model_additional_constraints'].items():
                print(f"{key}: {value}")
        
        return result
    else:
        print(f"Error: {response.status_code}")
        print(response.json())
        return None

def check_image_dimensions(image_path: str, model: str = "dall-e-3", provider: str = "openai"):
    """Check if an image meets the model's dimension requirements."""
    
    # Get model limits
    limits = check_image_generation_limits("", model, provider)
    if not limits:
        return False
    
    # Parse max resolution
    max_width, max_height = map(int, limits['model_max_resolution'].split('x'))
    
    # Check image dimensions
    with Image.open(image_path) as img:
        width, height = img.size
        within_limits = width <= max_width and height <= max_height
        
        print(f"\nImage Dimensions Check:")
        print(f"Image size: {width}x{height}")
        print(f"Maximum allowed: {max_width}x{max_height}")
        print(f"Within limits: {within_limits}")
        
        return within_limits

# Example usage
if __name__ == "__main__":
    # Example prompt
    prompt = "A beautiful sunset over mountains with a lake in the foreground"
    
    # Check with different image models
    check_image_generation_limits(prompt, model="dall-e-3", provider="openai")
    check_image_generation_limits(prompt, model="stable-diffusion-xl-1024-v1-0", provider="stability")
    
    # Example image dimension check (if you have an image file)
    # check_image_dimensions("path/to/your/image.jpg", model="dall-e-3", provider="openai")
