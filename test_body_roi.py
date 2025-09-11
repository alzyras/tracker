#!/usr/bin/env python3

import sys
import os
import cv2
import numpy as np

# Add the project root to the path so we can import uv_app
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Fix the import issue by adding the uv_app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uv_app'))

from core.detection import BodyDetector

def test_body_roi_method():
    """Test that the get_body_roi method exists and works."""
    print("Testing BodyDetector get_body_roi method...")
    
    # Create a body detector
    detector = BodyDetector()
    
    # Check if the method exists
    if hasattr(detector, 'get_body_roi'):
        print("✅ get_body_roi method exists")
        
        # Create a test frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test the method
        bbox = (100, 100, 200, 200)
        try:
            roi = detector.get_body_roi(frame, bbox)
            print(f"✅ get_body_roi works, returned ROI with shape: {roi.shape}")
        except Exception as e:
            print(f"❌ get_body_roi failed with error: {e}")
    else:
        print("❌ get_body_roi method does not exist")
        print(f"Available methods: {[method for method in dir(detector) if not method.startswith('_')]}")

if __name__ == "__main__":
    test_body_roi_method()