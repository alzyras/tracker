#!/usr/bin/env python3
# uv_app/cleanup_person_data.py

"""
Script to clean up person data that might be causing recognition issues.
"""

import os
import shutil
from uv_app.config import SAVE_DIR

def cleanup_person_data():
    """Clean up person data directories."""
    print(f"Cleaning up person data in {SAVE_DIR}...")
    
    if not os.path.exists(SAVE_DIR):
        print("Save directory doesn't exist, nothing to clean up.")
        return
    
    # List all person directories
    person_dirs = [d for d in os.listdir(SAVE_DIR) if d.startswith("person_")]
    
    if not person_dirs:
        print("No person data found, nothing to clean up.")
        return
    
    print(f"Found {len(person_dirs)} person directories:")
    for d in person_dirs:
        print(f"  - {d}")
    
    # Ask for confirmation
    response = input("\nDo you want to remove all person data? (y/N): ")
    if response.lower() in ['y', 'yes']:
        for d in person_dirs:
            person_path = os.path.join(SAVE_DIR, d)
            try:
                shutil.rmtree(person_path)
                print(f"Removed {d}")
            except Exception as e:
                print(f"Error removing {d}: {e}")
        
        print("Cleanup completed!")
    else:
        print("Cleanup cancelled.")

if __name__ == "__main__":
    cleanup_person_data()