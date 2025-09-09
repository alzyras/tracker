#!/usr/bin/env python3
# uv_app/examples/plugin_example.py

"""
Example application demonstrating the plugin system.
This shows how to use the raw tracking framework with plugins.
"""

import sys
import os
import cv2
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.tracking import PersonTracker
from plugins.emotion_plugin import EmotionPlugin, SimpleEmotionPlugin
from plugins.activity_plugin import ActivityPlugin, SimpleActivityPlugin, PoseActivityPlugin


def main():
    """Main example function."""
    print("🚀 Face Tracking with Plugin System Example")
    print("=" * 50)
    
    # Create tracker with all features enabled
    tracker = PersonTracker(
        enable_face=True,
        enable_body=True,
        enable_pose=True
    )
    
    # Register plugins
    print("\n📦 Registering plugins...")
    
    # Emotion plugins
    emotion_plugin = EmotionPlugin(update_interval_ms=2000)  # Every 2 seconds
    simple_emotion_plugin = SimpleEmotionPlugin(update_interval_ms=1000)  # Every 1 second
    
    # Activity plugins
    activity_plugin = ActivityPlugin(update_interval_ms=3000)  # Every 3 seconds
    simple_activity_plugin = SimpleActivityPlugin(update_interval_ms=2000)  # Every 2 seconds
    pose_activity_plugin = PoseActivityPlugin(update_interval_ms=1000)  # Every 1 second
    
    # Register all plugins
    tracker.register_plugin(emotion_plugin)
    tracker.register_plugin(simple_emotion_plugin)
    tracker.register_plugin(activity_plugin)
    tracker.register_plugin(simple_activity_plugin)
    tracker.register_plugin(pose_activity_plugin)
    
    print("✅ All plugins registered")
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera")
        return
    
    print("\n🎥 Starting tracking...")
    print("Press 'q' to quit, 'r' to show results, 'c' to clear results")
    
    frame_count = 0
    last_results_show = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            processed_frame = tracker.process_frame(frame)
            
            # Add info overlay
            frame_count += 1
            visible_people = tracker.get_visible_people()
            
            # Draw info on frame
            cv2.putText(processed_frame, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(processed_frame, f"Visible People: {len(visible_people)}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show plugin results for each person
            y_offset = 90
            for person in visible_people:
                person_name = person.name or f"Person {person.track_id}"
                cv2.putText(processed_frame, f"{person_name}:", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                y_offset += 20
                
                # Get results for this person
                results = tracker.get_plugin_results(person_id=person.track_id)
                
                for plugin_name, result in results.items():
                    if 'error' not in result:
                        if 'emotion' in result:
                            text = f"  {plugin_name}: {result['emotion']}"
                        elif 'activity' in result:
                            text = f"  {plugin_name}: {result['activity']}"
                        else:
                            text = f"  {plugin_name}: {str(result)[:30]}..."
                        
                        cv2.putText(processed_frame, text, (10, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                        y_offset += 15
                
                y_offset += 10
            
            # Display frame
            cv2.imshow("Face Tracking with Plugins", processed_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                show_detailed_results(tracker)
            elif key == ord('c'):
                if tracker.plugin_manager:
                    tracker.plugin_manager.clear_old_results()
                print("Results cleared")
    
    except KeyboardInterrupt:
        print("\n⏹️  Stopped by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        tracker.save_all_data()
        print("✅ Application stopped and data saved")


def show_detailed_results(tracker):
    """Show detailed plugin results."""
    print("\n📊 Detailed Plugin Results:")
    print("-" * 40)
    
    all_results = tracker.get_plugin_results()
    
    if not all_results:
        print("No results available")
        return
    
    for key, result in all_results.items():
        person_id = result.get('person_id', 'Unknown')
        plugin_name = result.get('plugin', 'Unknown')
        timestamp = result.get('timestamp', 0)
        
        print(f"\nPerson {person_id} - {plugin_name}:")
        print(f"  Timestamp: {timestamp}")
        
        if 'error' in result:
            print(f"  Error: {result['error']}")
        else:
            plugin_result = result.get('result', {})
            for k, v in plugin_result.items():
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
