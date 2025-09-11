# uv_app/app.py

import cv2
import argparse
from typing import Union, Optional
from core.tracking import PersonTracker
from ui.display import VideoDisplayManager
from utils.validators import ConfigValidator
from core.logging import setup_logging
from config import LOGGING_CONFIG, SAVE_DIR
from plugins import PLUGIN_REGISTRY, PluginManager
from config_loader import config_loader


class TrackerApp:
    """Main application class for the face tracking system."""
    
    def __init__(self, enable_face: bool = True, enable_body: bool = False, enable_pose: bool = False):
        # Ensure logging is configured early
        logger_manager = setup_logging(LOGGING_CONFIG)
        logger_manager.setup_file_logging(SAVE_DIR)
        self.tracker = PersonTracker(enable_face, enable_body, enable_pose)
        self.display = VideoDisplayManager("Face Tracker")
        self.running = False
        self._setup_plugins()
    
    def _setup_plugins(self):
        """Set up plugins based on configuration from Streamlit app."""
        # Load plugin configuration
        plugin_settings = config_loader.get_plugin_settings()
        
        # Register plugins based on configuration
        for plugin_name, plugin_class in PLUGIN_REGISTRY.items():
            # Check if plugin is enabled in configuration
            if config_loader.is_plugin_enabled(plugin_name):
                try:
                    # Get plugin parameters from configuration
                    plugin_config = plugin_settings.get(plugin_name, {})
                    
                    # Filter out the 'enabled' parameter and 'update_interval_ms' 
                    # which we'll handle separately
                    plugin_params = {
                        k: v for k, v in plugin_config.items() 
                        if k not in ['enabled', 'update_interval_ms']
                    }
                    
                    # Get update interval (default to 1000ms if not specified)
                    update_interval = plugin_config.get('update_interval_ms', 1000)
                    
                    # Create plugin instance with configured parameters
                    if plugin_params:
                        plugin = plugin_class(update_interval_ms=update_interval, **plugin_params)
                    else:
                        plugin = plugin_class(update_interval_ms=update_interval)
                    
                    # Register the plugin
                    self.tracker.plugin_manager.register_plugin(plugin)
                    print(f"Registered plugin: {plugin_name}")
                except Exception as e:
                    print(f"Error registering plugin {plugin_name}: {e}")
            else:
                print(f"Plugin {plugin_name} is disabled in configuration")
    
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
                
                # Display frame based on configuration
                display_mode = config_loader.get_ui_parameter('display_mode', 'opencv')
                
                if display_mode == 'opencv':
                    self.display.show_frame(processed_frame)
                elif display_mode == 'streamlit':
                    # For Streamlit mode, we would need to implement a different approach
                    # This would typically involve sending frames to a Streamlit component
                    self.display.show_frame(processed_frame)
                # If display_mode is 'none', we don't display anything
                
                # Check for exit (only if using OpenCV display)
                if display_mode in ['opencv', 'streamlit']:
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
