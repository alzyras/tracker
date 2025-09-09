# uv_app/plugins/base.py

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np


class BasePlugin(ABC):
    """Base class for all tracking plugins."""
    
    def __init__(self, name: str, update_interval_ms: int = 1000):
        """
        Initialize plugin.
        
        Args:
            name: Plugin name
            update_interval_ms: How often to update in milliseconds
        """
        self.name = name
        self.update_interval_ms = update_interval_ms
        self.last_update = 0
        self.enabled = True
    
    @abstractmethod
    def process_person(self, person, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a person and return results.
        
        Args:
            person: TrackedPerson object
            frame: Current frame
            
        Returns:
            Dictionary with processing results
        """
        pass
    
    def should_update(self, current_time_ms: int) -> bool:
        """Check if plugin should update based on interval."""
        if not self.enabled:
            return False
        return current_time_ms - self.last_update >= self.update_interval_ms
    
    def update_timestamp(self, current_time_ms: int) -> None:
        """Update last update timestamp."""
        self.last_update = current_time_ms
    
    def enable(self) -> None:
        """Enable the plugin."""
        self.enabled = True
    
    def disable(self) -> None:
        """Disable the plugin."""
        self.enabled = False


class FacePlugin(BasePlugin):
    """Base class for face-based plugins."""
    
    def process_person(self, person, frame: np.ndarray) -> Dict[str, Any]:
        """Process person using face image."""
        face_image = person.get_current_face_image()
        if face_image is None:
            return {"error": "No face image available"}
        
        return self.process_face(face_image, person)
    
    @abstractmethod
    def process_face(self, face_image: np.ndarray, person) -> Dict[str, Any]:
        """Process face image and return results."""
        pass


class BodyPlugin(BasePlugin):
    """Base class for body-based plugins."""
    
    def process_person(self, person, frame: np.ndarray) -> Dict[str, Any]:
        """Process person using body image."""
        body_image = person.get_current_body_image()
        if body_image is None:
            return {"error": "No body image available"}
        
        return self.process_body(body_image, person)
    
    @abstractmethod
    def process_body(self, body_image: np.ndarray, person) -> Dict[str, Any]:
        """Process body image and return results."""
        pass


class PosePlugin(BasePlugin):
    """Base class for pose-based plugins."""
    
    def process_person(self, person, frame: np.ndarray) -> Dict[str, Any]:
        """Process person using pose landmarks."""
        pose_landmarks = person.get_current_pose_landmarks()
        if pose_landmarks is None:
            return {"error": "No pose landmarks available"}
        
        return self.process_pose(pose_landmarks, person)
    
    @abstractmethod
    def process_pose(self, pose_landmarks: Any, person) -> Dict[str, Any]:
        """Process pose landmarks and return results."""
        pass
