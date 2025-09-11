# test_smolvlm_plugin.py

import sys
import os
import time
import cv2
import numpy as np

# Add the project root to the path so we can import uv_app
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from uv_app.plugins.smolvlm_plugin import SmolVLMPlugin


def test_smolvlm_plugin():
    """Test the SmolVLM plugin with a sample image."""
    # Load the actual test image
    if os.path.exists('tomas_phone.jpg'):
        body_image = cv2.imread('tomas_phone.jpg')
        print("Loaded test image: tomas_phone.jpg")
    else:
        # Create a sample body image (just a random image for testing)
        body_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        print("Created random test image")
    
    # Create a mock person object
    class MockPerson:
        def __init__(self, track_id):
            self.track_id = track_id
    
    person = MockPerson(track_id=1)
    
    # Create the plugin
    plugin = SmolVLMPlugin(
        api_url="http://localhost:9000/describe",
        update_interval_ms=1000
    )
    
    print("Testing SmolVLM plugin...")
    print("Sending body image to SmolVLM API...")
    
    # Process the body image
    result = plugin.process_body(body_image, person)
    print(f"Initial result: {result}")
    
    # Wait a bit to see if the async request completes
    print("Waiting for API response...")
    time.sleep(5)
    
    # Process again to see updated result
    result = plugin.process_body(body_image, person)
    print(f"Updated result: {result}")
    
    print("Test completed!")


if __name__ == "__main__":
    test_smolvlm_plugin()