import requests
import json

def check_text_limits(text: str, model: str = "gpt-4", provider: str = "openai", api_key: str = None):
    """Check token limits for text generation."""
    
    # API endpoint (assuming the service is running locally)
    url = "http://localhost:8000/check-limits"
    
    # Request payload
    payload = {
        "content": text,
        "model": model,
        "provider": provider,
        "model_type": "text",
        "api_key": api_key
    }
    
    # Make the request
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nResults for {model} ({provider}):")
        print(f"Total tokens: {result['total_tokens']}")
        print(f"Within limit: {result['is_within_limit']}")
        print(f"Model max tokens: {result['model_max_tokens']}")
        
        if not result['is_within_limit']:
            print("\nBatch processing recommended:")
            print(f"Recommended batch size: {result['recommended_batch_size']}")
            print("\nSuggested batches:")
            for batch in result['batches']:
                print(f"Batch {batch['batch']}: {batch['tokens']} tokens")
        
        return result
    else:
        print(f"Error: {response.status_code}")
        print(response.json())
        return None

# Example usage
if __name__ == "__main__":
    # Example text (a long article or document)
    long_text = """
    [Your long text here...]
    """ * 1000  # Making it artificially long for demonstration
    
    # Check with different models
    check_text_limits(long_text, model="gpt-4", provider="openai")
    check_text_limits(long_text, model="claude-3-opus-20240229", provider="anthropic")
    check_text_limits(long_text, model="gemini-pro", provider="google")
