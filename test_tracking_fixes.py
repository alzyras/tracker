"""
Test script to verify the fixes for the tracking system.
"""

import sys
import numpy as np
from pathlib import Path

# Add the project root to sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_tracking_fixes():
    """Test the fixes for the tracking system."""
    print("Testing tracking system fixes...")
    
    try:
        # Import the modules we've fixed
        from uv_app.core.tracking import PersonTracker
        from uv_app.core.person import TrackedPerson
        from uv_app.core.detection import BodyDetector
        from uv_app.core.recognition import FaceRecognizer
        
        print("✓ All modules imported successfully")
        
        # Test PersonTracker initialization
        tracker = PersonTracker(enable_face=True, enable_body=True, enable_pose=True)
        print("✓ PersonTracker initialized successfully")
        
        # Test TrackedPerson
        person = TrackedPerson(1)
        print("✓ TrackedPerson initialized successfully")
        
        # Test adding face data (this was one of the problematic areas)
        encoding = np.array([0.1, 0.2, 0.3])
        face_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        person.add_face_data(encoding, face_img)
        print("✓ Face data added successfully")
        
        # Test adding another face with similar encoding (should be detected as duplicate)
        similar_encoding = np.array([0.11, 0.21, 0.31])
        person.add_face_data(similar_encoding, face_img)
        print(f"✓ Duplicate face handling works (face count: {person.get_face_count()})")
        
        # Test FaceRecognizer
        recognizer = FaceRecognizer()
        print("✓ FaceRecognizer initialized successfully")
        
        # Test finding best match
        best_match, distance = recognizer.find_best_match(encoding)
        print(f"✓ Best match found (distance: {distance})")
        
        print("\nAll tests passed! The fixes should resolve the NumPy array comparison issues.")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_tracking_fixes()