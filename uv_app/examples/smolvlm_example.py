# uv_app/examples/smolvlm_example.py

import sys
import os
import time
import cv2

# Add the project root to the path so we can import uv_app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uv_app.core.tracking import PersonTracker
from uv_app.plugins.smolvlm_plugin import create_smolvlm_plugin


def main():
    """Example of using the SmolVLM plugin with the person tracker."""
    print("Initializing PersonTracker with SmolVLM plugin...")
    
    # Create tracker
    tracker = PersonTracker(enable_face=True, enable_body=True, enable_pose=True)
    
    # Create and register SmolVLM plugin
    smolvlm_plugin = create_smolvlm_plugin(
        api_url="http://localhost:9000/describe",
        update_interval_ms=2000
    )
    tracker.register_plugin(smolvlm_plugin)
    
    print("SmolVLM plugin registered!")
    print("Starting camera feed (press 'q' to quit)...")
    
    # Start camera
    cap = cv2.VideoCapture(0)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            processed_frame = tracker.process_frame(frame)
            
            # Display frame
            cv2.imshow("Person Tracker with SmolVLM", processed_frame)
            
            # Get plugin results
            results = tracker.get_plugin_results(plugin_name="smolvlm_activity")
            if results:
                print(f"SmolVLM Results: {results}")
            
            # Break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()