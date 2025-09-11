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


def test_smolvlm_with_real_body_detection():
    """Test the SmolVLM plugin with actual body detection."""
    print("Testing SmolVLM plugin with real body detection...")
    
    # Create plugin manager
    plugin_manager = PluginManager()
    
    # Create and register SmolVLM plugin with fast interval for testing
    smolvlm_plugin = SmolVLMPlugin(
        api_url="http://localhost:9000/describe",
        update_interval_ms=2000  # 2 seconds for faster testing
    )
    plugin_manager.register_plugin(smolvlm_plugin)
    
    # Create a mock person object that simulates having body data
    class MockPerson:
        def __init__(self, track_id):
            self.track_id = track_id
            self.is_visible = True
            self.name = None
            # Load actual body image if available
            if os.path.exists('tomas_phone.jpg'):
                self._body_image = cv2.imread('tomas_phone.jpg')
                print(f"Loaded body image with shape: {self._body_image.shape}")
            else:
                # Create a sample body image
                self._body_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                print("Created sample body image")
        
        def get_current_body_image(self):
            return self._body_image
    
    person = MockPerson(track_id=1)
    
    print("Processing person with SmolVLM plugin...")
    
    # Process multiple times to test the interval logic
    for i in range(3):
        print(f"Iteration {i+1}:")
        
        # Reset the plugin's last update time to force processing
        smolvlm_plugin.last_update = 0
        
        # Process the person
        plugin_manager.process_people([person], person.get_current_body_image())
        
        print("  Waiting for processing...")
        time.sleep(3)  # Wait for API request to complete
    
    print("Test completed!")


if __name__ == "__main__":
    test_smolvlm_with_real_body_detection()