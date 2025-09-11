#!/usr/bin/env python3

import sys
import os
import time
import cv2
import numpy as np

# Add the project root to the path so we can import uv_app
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from uv_app.plugins.smolvlm_plugin import SmolVLMPlugin


def test_smolvlm_direct():
    """Test the SmolVLM plugin directly."""
    print("Testing SmolVLM plugin directly...")
    
    # Create and register SmolVLM plugin
    smolvlm_plugin = SmolVLMPlugin(
        api_url="http://localhost:9000/describe",
        update_interval_ms=1000  # Fast interval for testing
    )
    
    # Create a mock person object
    class MockPerson:
        def __init__(self, track_id):
            self.track_id = track_id
            self.is_visible = True
            self.name = None
    
    person = MockPerson(track_id=1)
    
    # Load the actual test image
    if os.path.exists('tomas_phone.jpg'):
        body_image = cv2.imread('tomas_phone.jpg')
        print("Loaded test image: tomas_phone.jpg")
    else:
        # Create a sample body image (just a random image for testing)
        body_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        print("Created random test image")
    
    # Update person with body image
    person.get_current_body_image = lambda: body_image
    
    print("Calling process_person directly...")
    
    # Call process_person directly to bypass timing checks
    result = smolvlm_plugin.process_person(person, body_image)
    print(f"Initial result: {result}")
    
    print("Waiting for API response...")
    time.sleep(5)  # Wait for the API request to complete
    
    # Call again to get the updated result
    result = smolvlm_plugin.process_person(person, body_image)
    print(f"Updated result: {result}")
    
    print("Test completed!")


if __name__ == "__main__":
    test_smolvlm_direct()