# uv_app/utils/validators.py

import os
import cv2
import numpy as np
from typing import Any, List, Tuple, Optional, Union


class ConfigValidator:
    """Validates configuration parameters."""
    
    @staticmethod
    def validate_threshold(value: float, min_val: float = 0.0, max_val: float = 1.0) -> bool:
        """Validate threshold values."""
        return isinstance(value, (int, float)) and min_val <= value <= max_val
    
    @staticmethod
    def validate_positive_int(value: Any) -> bool:
        """Validate positive integer values."""
        return isinstance(value, int) and value > 0
    
    @staticmethod
    def validate_directory_path(path: str) -> bool:
        """Validate directory path."""
        return isinstance(path, str) and len(path) > 0
    
    @staticmethod
    def validate_video_source(source: Union[int, str]) -> bool:
        """Validate video source (webcam index or file path)."""
        if isinstance(source, int):
            return source >= 0
        elif isinstance(source, str):
            return os.path.exists(source) or source.startswith(('http://', 'https://', 'rtsp://'))
        return False


class ImageValidator:
    """Validates image data and operations."""
    
    @staticmethod
    def validate_frame(frame: Any) -> bool:
        """Validate if object is a valid OpenCV frame."""
        return isinstance(frame, np.ndarray) and len(frame.shape) == 3 and frame.shape[2] == 3
    
    @staticmethod
    def validate_face_encoding(encoding: Any) -> bool:
        """Validate face encoding."""
        return (isinstance(encoding, np.ndarray) and 
                len(encoding.shape) == 1 and 
                encoding.dtype in (np.float32, np.float64))
    
    @staticmethod
    def validate_bbox(bbox: Any, frame_shape: Optional[Tuple] = None) -> bool:
        """Validate bounding box format and coordinates."""
        if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
            return False
        
        try:
            top, right, bottom, left = bbox
            if not all(isinstance(x, int) for x in bbox):
                return False
            
            if not (top < bottom and left < right):
                return False
            
            if frame_shape is not None:
                height, width = frame_shape[:2]
                if not (0 <= top < bottom <= height and 0 <= left < right <= width):
                    return False
            
            return True
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_face_locations(locations: Any) -> bool:
        """Validate face locations list."""
        if not isinstance(locations, list):
            return False
        
        for location in locations:
            if not ImageValidator.validate_bbox(location):
                return False
        
        return True


class PersonValidator:
    """Validates person-related data."""
    
    @staticmethod
    def validate_track_id(track_id: Any) -> bool:
        """Validate track ID."""
        return isinstance(track_id, int) and track_id > 0
    
    @staticmethod
    def validate_person_name(name: Any) -> bool:
        """Validate person name."""
        return isinstance(name, str) and len(name.strip()) > 0
    
    @staticmethod
    def validate_face_images(images: Any) -> bool:
        """Validate face images list."""
        if not isinstance(images, list):
            return False
        
        for img in images:
            if not ImageValidator.validate_frame(img):
                return False
        
        return True
    
    @staticmethod
    def validate_emotion(emotion: Any) -> bool:
        """Validate emotion string."""
        valid_emotions = ['happy', 'sad', 'angry', 'surprised', 'fearful', 'disgusted', 'neutral']
        return isinstance(emotion, str) and emotion.lower() in valid_emotions


class FileValidator:
    """Validates file operations and paths."""
    
    @staticmethod
    def validate_image_file(file_path: str) -> bool:
        """Validate image file exists and is readable."""
        if not os.path.exists(file_path):
            return False
        
        try:
            img = cv2.imread(file_path)
            return img is not None
        except Exception:
            return False
    
    @staticmethod
    def validate_json_file(file_path: str) -> bool:
        """Validate JSON file exists and is readable."""
        if not os.path.exists(file_path):
            return False
        
        try:
            import json
            with open(file_path, 'r') as f:
                json.load(f)
            return True
        except Exception:
            return False
    
    @staticmethod
    def validate_numpy_file(file_path: str) -> bool:
        """Validate NumPy file exists and is readable."""
        if not os.path.exists(file_path):
            return False
        
        try:
            np.load(file_path)
            return True
        except Exception:
            return False
    
    @staticmethod
    def validate_directory_writable(dir_path: str) -> bool:
        """Validate directory exists and is writable."""
        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path, exist_ok=True)
            except Exception:
                return False
        
        if not os.path.isdir(dir_path):
            return False
        
        try:
            test_file = os.path.join(dir_path, '.write_test')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            return True
        except Exception:
            return False


class DetectionValidator:
    """Validates detection and recognition results."""
    
    @staticmethod
    def validate_detection_results(locations: List, encodings: List) -> bool:
        """Validate that detection results are consistent."""
        if not isinstance(locations, list) or not isinstance(encodings, list):
            return False
        
        if len(locations) != len(encodings):
            return False
        
        for location, encoding in zip(locations, encodings):
            if not ImageValidator.validate_bbox(location):
                return False
            if not ImageValidator.validate_face_encoding(encoding):
                return False
        
        return True
    
    @staticmethod
    def validate_candidate_face(candidate: dict) -> bool:
        """Validate candidate face dictionary."""
        required_keys = ['encoding', 'img', 'bbox', 'count']
        if not all(key in candidate for key in required_keys):
            return False
        
        if not ImageValidator.validate_face_encoding(candidate['encoding']):
            return False
        
        if not ImageValidator.validate_frame(candidate['img']):
            return False
        
        if not ImageValidator.validate_bbox(candidate['bbox']):
            return False
        
        if not isinstance(candidate['count'], int) or candidate['count'] < 0:
            return False
        
        return True
