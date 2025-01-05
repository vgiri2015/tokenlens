import requests
import json
from moviepy.editor import VideoFileClip

def check_video_generation_limits(prompt: str, model: str = "make-a-video", provider: str = "meta", api_key: str = None):
    """Check limits for video generation."""
    
    url = "http://localhost:8000/check-limits"
    
    payload = {
        "content": prompt,
        "model": model,
        "provider": provider,
        "model_type": "video",
        "api_key": api_key
    }
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nVideo Generation Limits for {model} ({provider}):")
        print(f"Maximum Duration: {result['model_max_duration']} seconds")
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

def check_video_constraints(video_path: str, model: str = "make-a-video", provider: str = "meta"):
    """Check if a video meets the model's constraints."""
    
    # Get model limits
    limits = check_video_generation_limits("", model, provider)
    if not limits:
        return False
    
    # Check video properties
    with VideoFileClip(video_path) as video:
        duration = video.duration
        fps = video.fps
        width, height = video.size
        
        max_duration = float(limits['model_max_duration'])
        within_duration = duration <= max_duration
        
        print(f"\nVideo Constraints Check:")
        print(f"Duration: {duration:.2f}s (max: {max_duration}s)")
        print(f"FPS: {fps}")
        print(f"Resolution: {width}x{height}")
        print(f"Within duration limit: {within_duration}")
        
        return within_duration

# Example usage
if __name__ == "__main__":
    # Example prompt
    prompt = "A timelapse of a flower blooming in a garden"
    
    # Check with different video models
    check_video_generation_limits(prompt, model="make-a-video", provider="meta")
    check_video_generation_limits(prompt, model="stable-video-diffusion", provider="stability")
    
    # Example video constraint check (if you have a video file)
    # check_video_constraints("path/to/your/video.mp4", model="make-a-video", provider="meta")
