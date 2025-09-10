# uv_app/app.py

import cv2
import argparse
from typing import Union, Optional
from core.tracking import PersonTracker
from ui.display import VideoDisplayManager
from utils.validators import ConfigValidator
from core.logging import setup_logging
from config import LOGGING_CONFIG, SAVE_DIR


class TrackerApp:
    """Main application class for the face tracking system."""
    
    def __init__(self, enable_face: bool = True, enable_body: bool = False, enable_pose: bool = False):
        # Ensure logging is configured early
        logger_manager = setup_logging(LOGGING_CONFIG)
        logger_manager.setup_file_logging(SAVE_DIR)
        self.tracker = PersonTracker(enable_face, enable_body, enable_pose)
        self.display = VideoDisplayManager("Face Tracker")
        self.running = False
    
    def run(self, video_source: Union[int, str] = 0) -> None:
        """
        Run the tracking application.
        
        Args:
            video_source: Video source (webcam index, file path, or stream URL)
        """
        if not ConfigValidator.validate_video_source(video_source):
            raise ValueError(f"Invalid video source: {video_source}")
        
        # Initialize video capture
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source {video_source}")
        
        print(f"Starting tracker with video source: {video_source}")
        print("Press 'q' or ESC to quit")
        
        self.running = True
        
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    print("End of video stream")
                    break
                
                # Process frame
                processed_frame = self.tracker.process_frame(frame)
                
                # Display frame
                self.display.show_frame(processed_frame)
                
                # Check for exit
                key = self.display.wait_for_key(1)
                if self.display.should_exit(key):
                    break
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            # Cleanup
            self.cleanup(cap)
    
    def cleanup(self, cap: cv2.VideoCapture) -> None:
        """Clean up resources."""
        self.running = False
        cap.release()
        self.display.destroy_windows()
        self.tracker.save_all_data()
        print("Application stopped and data saved")
    
    def get_tracked_people_count(self) -> int:
        """Get number of currently tracked people."""
        return len(self.tracker.get_tracked_people())
    
    def get_lost_people_count(self) -> int:
        """Get number of lost people."""
        return len(self.tracker.get_lost_people())


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="Face Tracking Application")
    parser.add_argument("--source", "-s", default=0, 
                       help="Video source (webcam index, file path, or stream URL)")
    parser.add_argument("--no-face", action="store_true", 
                       help="Disable face detection")
    parser.add_argument("--enable-body", action="store_true", 
                       help="Enable body detection")
    parser.add_argument("--enable-pose", action="store_true", 
                       help="Enable pose detection")
    
    args = parser.parse_args()
    
    # Convert source to appropriate type
    try:
        video_source = int(args.source)
    except ValueError:
        video_source = args.source
    
    # Create and run application
    app = TrackerApp(
        enable_face=not args.no_face,
        enable_body=args.enable_body,
        enable_pose=args.enable_pose
    )
    
    try:
        app.run(video_source)
    except Exception as e:
        print(f"Error running application: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
