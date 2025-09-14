#!/usr/bin/env python3
"""
Video Stream Processor
======================

This script allows you to select an MP4 video file and use it as a camera stream
with face detection, tracking, and overlays.

Usage:
    python video_stream_processor.py

The script will open a file dialog to select an MP4 file, then process it
with the same detection and tracking capabilities as the main application.
"""

import cv2
import tkinter as tk
from tkinter import filedialog
import sys
import os
from pathlib import Path

# Add the uv_app directory to the Python path
sys.path.append(str(Path(__file__).parent / "uv_app"))

from core.tracking import PersonTracker
from ui.display import VideoDisplayManager


def select_video_file():
    """Open a file dialog to select an MP4 video file."""
    # Create a root window and hide it
    root = tk.Tk()
    root.withdraw()
    
    # Show file dialog
    file_path = filedialog.askopenfilename(
        title="Select MP4 Video File",
        filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
    )
    
    # Destroy the root window
    root.destroy()
    
    return file_path


def main():
    """Main function to process video file as camera stream."""
    print("Video Stream Processor")
    print("======================")
    
    # Select video file
    video_path = select_video_file()
    
    if not video_path:
        print("No file selected. Exiting.")
        return 1
    
    if not os.path.exists(video_path):
        print(f"File not found: {video_path}")
        return 1
    
    print(f"Processing video: {video_path}")
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return 1
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video properties: {width}x{height} @ {fps} FPS")
    
    # Initialize tracker and display manager
    tracker = PersonTracker(enable_face=True, enable_body=True, enable_pose=True)
    display = VideoDisplayManager("Video Stream Processor")
    
    print("Starting video processing...")
    print("Press 'q' or ESC to quit")
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            
            if not ret:
                print("End of video stream")
                # Reset to beginning for continuous playback
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # Process frame with tracker
            processed_frame = tracker.process_frame(frame)
            
            # Display frame
            display.show_frame(processed_frame)
            
            # Check for exit key
            key = display.wait_for_key(1)
            if display.should_exit(key):
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error during processing: {e}")
        return 1
    finally:
        # Cleanup
        cap.release()
        try:
            display.destroy_windows()
        except:
            # Ignore errors during window destruction
            pass
        print("Video processing stopped")
    
    return 0


if __name__ == "__main__":
    exit(main())