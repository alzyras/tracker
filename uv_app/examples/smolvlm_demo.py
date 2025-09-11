#!/usr/bin/env python3
"""
Example script demonstrating how to use the SmolVLM plugin with the person tracker.
This script shows how to:
1. Initialize the person tracker
2. Register the SmolVLM plugin
3. Process frames and get activity descriptions
"""

import sys
import os
import time
import cv2

# Add the project root to the path so we can import uv_app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uv_app.core.tracking import PersonTracker
from uv_app.plugins.smolvlm_plugin import create_smolvlm_plugin


def main():
    """Main function to demonstrate the SmolVLM plugin."""
    print("🚀 Initializing Person Tracker with SmolVLM Plugin")
    print("=" * 50)
    
    # Create tracker with all features enabled
    tracker = PersonTracker(enable_face=True, enable_body=True, enable_pose=True)
    
    # Create and register SmolVLM plugin (5 second interval)
    smolvlm_plugin = create_smolvlm_plugin(
        api_url="http://localhost:9000/describe",
        update_interval_ms=5000  # Update every 5 seconds
    )
    tracker.register_plugin(smolvlm_plugin)
    
    print("✅ SmolVLM plugin registered!")
    print("📝 The plugin will:")
    print("   - Capture body images of detected people")
    print("   - Send them to the SmolVLM API for activity detection every 5 seconds")
    print("   - Log what each person is doing in the format:")
    print("     'Person ID X is doing: {activity description}'")
    print("   - Return results without blocking the main thread")
    print()
    
    # Try to open camera (if available)
    print("📹 Starting camera feed...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("⚠️  Camera not available. Showing how to use with existing images...")
        print()
        
        # Show how to get results from the plugin
        print("💡 To see the plugin in action:")
        print("   1. Make sure the SmolVLM API is running on http://localhost:9000")
        print("   2. Run this script with a camera connected")
        print("   3. The plugin will automatically detect and describe activities every 5 seconds")
        print()
        print("📝 Example log output you'll see:")
        print("   Person ID 1 is doing: The person is sitting at a desk working on a computer")
        print("   Person ID 2 is doing: The person is walking across the room")
        print()
        print("✅ Plugin implementation complete!")
        return
    
    print("📸 Camera started! Press 'q' to quit.")
    print("📝 Watch the console for activity logs...")
    print()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame with person tracker
            processed_frame = tracker.process_frame(frame)
            
            # Display frame
            cv2.imshow("Person Tracker with SmolVLM", processed_frame)
            
            # Break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n⏹️  Stopping...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    print("✅ Demo completed!")


if __name__ == "__main__":
    main()