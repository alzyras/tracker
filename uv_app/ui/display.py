# uv_app/ui/display.py

import cv2
import numpy as np
from typing import Tuple, Optional


class FaceDisplayManager:
    """Handles face bounding box display and labeling."""
    
    @staticmethod
    def draw_face_box(frame: np.ndarray, box: Tuple[int, int, int, int], 
                     label: str, color: Tuple[int, int, int] = (0, 255, 0)) -> None:
        """
        Draw bounding box and label on frame.
        
        Args:
            frame: Input frame
            box: Face location tuple (top, right, bottom, left)
            label: Text label to display
            color: BGR color tuple
        """
        top, right, bottom, left = box
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    @staticmethod
    def draw_face_box_with_emotion(frame: np.ndarray, box: Tuple[int, int, int, int], 
                                  label: str, emotion: Optional[str] = None,
                                  color: Tuple[int, int, int] = (0, 255, 0)) -> None:
        """
        Draw bounding box, label, and emotion on frame.
        
        Args:
            frame: Input frame
            box: Face location tuple (top, right, bottom, left)
            label: Text label to display
            emotion: Emotion text to display below the box
            color: BGR color tuple
        """
        top, right, bottom, left = box
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw emotion below the box if available
        if emotion:
            emotion_text = f"Emotion: {emotion}"
            cv2.putText(frame, emotion_text, (left, bottom + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    @staticmethod
    def draw_body_box(frame: np.ndarray, box: Tuple[int, int, int, int], 
                     label: Optional[str] = None, color: Tuple[int, int, int] = (255, 0, 0)) -> None:
        """
        Draw body bounding box on frame.
        
        Args:
            frame: Input frame
            box: Body location tuple (x1, y1, x2, y2)
            label: Optional text label
            color: BGR color tuple
        """
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if label:
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    @staticmethod
    def draw_info_panel(frame: np.ndarray, info: dict) -> None:
        """
        Draw information panel on frame.
        
        Args:
            frame: Input frame
            info: Dictionary of information to display
        """
        y_offset = 30
        for key, value in info.items():
            text = f"{key}: {value}"
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25


class VideoDisplayManager:
    """Handles video display and window management."""
    
    def __init__(self, window_name: str = "Tracker"):
        self.window_name = window_name
        self.is_window_created = False
    
    def create_window(self) -> None:
        """Create the display window."""
        if not self.is_window_created:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            self.is_window_created = True
    
    def show_frame(self, frame: np.ndarray) -> None:
        """
        Display a frame in the window.
        
        Args:
            frame: Frame to display
        """
        self.create_window()
        cv2.imshow(self.window_name, frame)
    
    def wait_for_key(self, timeout: int = 1) -> int:
        """
        Wait for a key press.
        
        Args:
            timeout: Timeout in milliseconds
            
        Returns:
            Key code or -1 if timeout
        """
        return cv2.waitKey(timeout) & 0xFF
    
    def should_exit(self, key: int) -> bool:
        """
        Check if the application should exit.
        
        Args:
            key: Key code from waitKey
            
        Returns:
            True if should exit
        """
        return key in (27, ord('q'))  # ESC or 'q'
    
    def destroy_windows(self) -> None:
        """Destroy all OpenCV windows."""
        cv2.destroyAllWindows()
    
    def resize_frame(self, frame: np.ndarray, max_size: int = 640) -> np.ndarray:
        """
        Resize frame if larger than max size.
        
        Args:
            frame: Input frame
            max_size: Maximum dimension size
            
        Returns:
            Resized frame
        """
        h, w = frame.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            return cv2.resize(frame, (int(w * scale), int(h * scale)))
        return frame
