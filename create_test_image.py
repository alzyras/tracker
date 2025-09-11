#!/usr/bin/env python3

import sys
import os
import base64
import cv2
import numpy as np

# Add the project root to the path so we can import uv_app
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_api_with_valid_image():
    """Test the API with a valid image."""
    # Create a simple test image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[50:70, 30:70] = [255, 255, 255]  # White rectangle
    
    # Encode as base64
    _, buffer = cv2.imencode('.jpg', img)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # Save to file for testing
    with open('test_image.jpg', 'wb') as f:
        f.write(buffer)
    
    print(f"Image encoded as base64 (first 50 chars): {image_base64[:50]}...")
    print(f"Image size: {len(image_base64)} characters")
    
    # Test with curl
    import json
    payload = {
        'image_b64': image_base64,
        'max_new_tokens': 50
    }
    
    with open('test_payload.json', 'w') as f:
        json.dump(payload, f)
    
    print("Created test_payload.json")
    print("You can test with:")
    print(f"curl -X POST http://localhost:9000/describe -H 'Content-Type: application/json' -d @test_payload.json")

if __name__ == "__main__":
    test_api_with_valid_image()