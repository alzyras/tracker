#!/usr/bin/env python3

import sys
import os

# Add the project root and uv_app to the path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'uv_app'))

import cv2
import numpy as np
from core.detection import BodyDetector, get_body_roi

def test_method_exists():
    """Test if the get_body_roi function exists."""
    # Test the standalone function
    try:
        # Test with a sample frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = (100, 100, 300, 300)
        
        roi = get_body_roi(frame, bbox)
        print(f"✅ get_body_roi function works, ROI shape: {roi.shape}")
    except Exception as e:
        print(f"❌ get_body_roi function failed: {e}")
        return

if __name__ == "__main__":
    test_method_exists()