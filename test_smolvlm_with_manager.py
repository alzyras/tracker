#!/usr/bin/env python3

import sys
import os
import time
import cv2
import numpy as np

# Add the project root to the path so we can import uv_app
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from uv_app.plugins.manager import PluginManager
from uv_app.plugins.smolvlm_plugin import SmolVLMPlugin


def test_smolvlm_with_plugin_manager():
    """Test the SmolVLM plugin with the plugin manager to verify logging."""
    print("Testing SmolVLM plugin with Plugin Manager...")
    
    # Create plugin manager
    plugin_manager = PluginManager()
    
    # Create and register SmolVLM plugin
    smolvlm_plugin = SmolVLMPlugin(
        api_url="http://localhost:9000/describe",
        update_interval_ms=1000  # Fast interval for testing
    )
    plugin_manager.register_plugin(smolvlm_plugin)
    
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
    
    print("First processing call (should start async API request)...")
    
    # Manually set the last update time to 0 so the plugin will process
    smolvlm_plugin.last_update = 0
    
    # First call - starts the async API request
    plugin_manager.process_people([person], body_image)
    
    print("Waiting for API response...")
    time.sleep(5)  # Wait for the API request to complete
    
    print("Second processing call (should get final result)...")
    
    # Second call - should get the final result
    # Manually set the last update time to 0 so the plugin will process again
    smolvlm_plugin.last_update = 0
    plugin_manager.process_people([person], body_image)
    
    print("Waiting a bit more...")
    time.sleep(1)  # Give a moment for any final processing
    
    print("Test completed!")


if __name__ == "__main__":
    test_smolvlm_with_plugin_manager()