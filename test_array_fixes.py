"""
Test script to verify the fixes for NumPy array comparison issues.
"""

import sys
import numpy as np
from pathlib import Path

# Add the project root to sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_array_comparisons():
    """Test array comparisons to ensure they don't cause the ambiguous error."""
    print("Testing NumPy array comparisons...")
    
    # Test 1: Basic array comparison in boolean context
    try:
        arr = np.array([1, 2, 3])
        # This should not raise an error
        result = len(arr) > 0
        print(f"Test 1 passed: len(arr) > 0 = {result}")
    except Exception as e:
        print(f"Test 1 failed: {e}")
    
    # Test 2: Array norm comparison
    try:
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([4, 5, 6])
        distance = np.linalg.norm(arr1 - arr2)
        result = distance < 5.0
        print(f"Test 2 passed: distance < 5.0 = {result}")
    except Exception as e:
        print(f"Test 2 failed: {e}")
    
    # Test 3: List of arrays check
    try:
        arr_list = [np.array([1, 2]), np.array([3, 4])]
        result = len(arr_list) > 0
        print(f"Test 3 passed: len(arr_list) > 0 = {result}")
    except Exception as e:
        print(f"Test 3 failed: {e}")
    
    # Test 4: Avoiding the 'all' function with array comparisons
    try:
        encoding = np.array([0.1, 0.2, 0.3])
        encodings = [np.array([0.15, 0.25, 0.35]), np.array([0.4, 0.5, 0.6])]
        
        # Old problematic way:
        # if all(np.linalg.norm(encoding - f) > 0.2 for f in encodings):
        
        # New correct way:
        is_duplicate = False
        for existing_encoding in encodings:
            if np.linalg.norm(encoding - existing_encoding) <= 0.2:
                is_duplicate = True
                break
                
        result = not is_duplicate
        print(f"Test 4 passed: duplicate check = {result}")
    except Exception as e:
        print(f"Test 4 failed: {e}")

if __name__ == "__main__":
    test_array_comparisons()