# uv_app/utils/helpers.py

import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from config import RESIZE_MAX


def resize_frame(frame: np.ndarray, max_size: int = RESIZE_MAX) -> np.ndarray:
    """
    Resize frame if larger than max allowed size.
    
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


def calculate_certainty_percentage(distance: float) -> float:
    """
    Calculate certainty percentage from face recognition distance.
    
    Args:
        distance: Face recognition distance (0-1)
        
    Returns:
        Certainty percentage (0-100)
    """
    return max(0, min(100, (1 - distance) * 100))


def validate_bbox(bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, int, int]) -> bool:
    """
    Validate bounding box coordinates.
    
    Args:
        bbox: Bounding box tuple (top, right, bottom, left)
        frame_shape: Frame shape (height, width, channels)
        
    Returns:
        True if valid, False otherwise
    """
    if len(bbox) != 4:
        return False
    
    top, right, bottom, left = bbox
    height, width = frame_shape[:2]
    
    return (0 <= top < bottom <= height and 
            0 <= left < right <= width and
            top < bottom and left < right)


def crop_face_from_frame(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    """
    Crop face region from frame using bounding box.
    
    Args:
        frame: Input frame
        bbox: Face bounding box (top, right, bottom, left)
        
    Returns:
        Cropped face image or None if invalid
    """
    if not validate_bbox(bbox, frame.shape):
        return None
    
    top, right, bottom, left = bbox
    return frame[top:bottom, left:right]


def calculate_face_center(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """
    Calculate center point of face bounding box.
    
    Args:
        bbox: Face bounding box (top, right, bottom, left)
        
    Returns:
        Center point (x, y)
    """
    top, right, bottom, left = bbox
    center_x = (left + right) // 2
    center_y = (top + bottom) // 2
    return center_x, center_y


def calculate_bbox_area(bbox: Tuple[int, int, int, int]) -> int:
    """
    Calculate area of bounding box.
    
    Args:
        bbox: Bounding box (top, right, bottom, left)
        
    Returns:
        Area in pixels
    """
    top, right, bottom, left = bbox
    return (bottom - top) * (right - left)


def normalize_encoding(encoding: np.ndarray) -> np.ndarray:
    """
    Normalize face encoding vector.
    
    Args:
        encoding: Face encoding vector
        
    Returns:
        Normalized encoding
    """
    norm = np.linalg.norm(encoding)
    if norm == 0:
        return encoding
    return encoding / norm


def calculate_encoding_distance(encoding1: np.ndarray, encoding2: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two face encodings.
    
    Args:
        encoding1: First face encoding
        encoding2: Second face encoding
        
    Returns:
        Distance between encodings
    """
    return np.linalg.norm(encoding1 - encoding2)


def filter_small_faces(face_locations: List[Tuple], min_area: int = 1000) -> List[Tuple]:
    """
    Filter out faces smaller than minimum area.
    
    Args:
        face_locations: List of face bounding boxes
        min_area: Minimum face area in pixels
        
    Returns:
        Filtered list of face bounding boxes
    """
    filtered = []
    for bbox in face_locations:
        if calculate_bbox_area(bbox) >= min_area:
            filtered.append(bbox)
    return filtered


def create_face_encoding_dict(encodings: List[np.ndarray], 
                             locations: List[Tuple]) -> List[Dict[str, Any]]:
    """
    Create list of dictionaries with face data.
    
    Args:
        encodings: List of face encodings
        locations: List of face locations
        
    Returns:
        List of dictionaries with face data
    """
    face_data = []
    for encoding, location in zip(encodings, locations):
        face_data.append({
            'encoding': encoding,
            'location': location,
            'center': calculate_face_center(location),
            'area': calculate_bbox_area(location)
        })
    return face_data


def format_time_elapsed(seconds: float) -> str:
    """
    Format elapsed time in a human-readable format.
    
    Args:
        seconds: Elapsed time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def safe_filename(filename: str) -> str:
    """
    Create a safe filename by removing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Safe filename
    """
    import re
    # Remove or replace invalid characters
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove multiple underscores
    safe_name = re.sub(r'_+', '_', safe_name)
    # Remove leading/trailing underscores
    safe_name = safe_name.strip('_')
    return safe_name
