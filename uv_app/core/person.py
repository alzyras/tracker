# uv_app/core/person.py

import os
import json
import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict, Any
from .logging import get_logger

# Import config differently to avoid import issues
try:
    from ..config import SAVE_DIR, MAX_FACE_IMAGES
except ImportError:
    # Fallback values if config can't be imported
    SAVE_DIR = "tracked_people"
    MAX_FACE_IMAGES = 30

logger = get_logger()


class TrackedPerson:
    """
    Represents a person being tracked.
    Stores face encodings, face/body bounding boxes, pose, and current state.
    """

    def __init__(self, track_id: int):
        self.track_id = track_id
        self.face_encodings: List[np.ndarray] = []
        self.face_images: List[np.ndarray] = []
        self.face_boxes: List[Tuple] = []  # (top, right, bottom, left)
        self.body_boxes: List[Tuple] = []  # (x1, y1, x2, y2)
        self.pose_landmarks: List[Any] = []  # list of pose objects
        self.missed_frames: int = 0
        self.name: Optional[str] = None
        self.mean_encoding: Optional[np.ndarray] = None
        
        # Current state (updated each frame)
        self.current_face_image: Optional[np.ndarray] = None
        self.current_body_image: Optional[np.ndarray] = None
        self.current_pose_landmarks: Optional[Any] = None
        self.current_face_bbox: Optional[Tuple] = None
        self.current_body_bbox: Optional[Tuple] = None
        self.is_visible: bool = False
        
        self.load_data()
        logger.debug(f"Initialized TrackedPerson with ID {track_id}")

    def update_current_state(self, face_image: Optional[np.ndarray] = None, 
                           body_image: Optional[np.ndarray] = None,
                           pose_landmarks: Optional[Any] = None,
                           face_bbox: Optional[Tuple] = None,
                           body_bbox: Optional[Tuple] = None) -> None:
        """Update current state with latest frame data."""
        if face_image is not None:
            self.current_face_image = face_image
        if body_image is not None:
            self.current_body_image = body_image
        if pose_landmarks is not None:
            self.current_pose_landmarks = pose_landmarks
        if face_bbox is not None:
            self.current_face_bbox = face_bbox
        if body_bbox is not None:
            self.current_body_bbox = body_bbox
        
        self.is_visible = True

    def save_face_image(self, face_img: np.ndarray) -> None:
        """Save individual face image to person's folder."""
        try:
            from ..config import SAVE_DIR
        except ImportError:
            SAVE_DIR = "tracked_people"
            
        person_dir = os.path.join(SAVE_DIR, f"person_{self.track_id}")
        os.makedirs(person_dir, exist_ok=True)
        
        # Count existing face images
        existing_faces = [f for f in os.listdir(person_dir) if f.startswith('face_') and f.endswith('.jpg')]
        face_num = len(existing_faces) + 1
        
        filename = f"face_{face_num}.jpg"
        cv2.imwrite(os.path.join(person_dir, filename), face_img)

    def update(self) -> None:
        """Reset missed frames counter."""
        self.missed_frames = 0

    def add_face_data(self, encoding: np.ndarray, face_img: np.ndarray, bbox: Optional[Tuple] = None) -> None:
        """Add a new face encoding/image and optional face bbox."""
        try:
            from ..config import MAX_FACE_IMAGES
        except ImportError:
            MAX_FACE_IMAGES = 30
            
        if len(self.face_encodings) >= MAX_FACE_IMAGES:
            return

        # Avoid duplicate encodings with a stricter threshold
        is_duplicate = False
        for existing_encoding in self.face_encodings:
            if np.linalg.norm(encoding - existing_encoding) <= 0.2:
                is_duplicate = True
                break
                
        if not is_duplicate:
            self.face_encodings.append(encoding)
            self.face_images.append(face_img)
            if bbox:
                self.face_boxes.append(bbox)
            self.update_mean_encoding()
            
            # Save face image immediately
            self.save_face_image(face_img)

    def add_body_data(self, body_bbox: Optional[Tuple] = None, pose: Optional[Any] = None, 
                     body_img: Optional[np.ndarray] = None) -> None:
        """Store body bbox, pose, and optional body image."""
        if body_bbox:
            self.body_boxes.append(body_bbox)
        if pose:
            self.pose_landmarks.append(pose)
        # Store body image if provided
        if body_img is not None:
            # You could also store body_img in memory if needed
            pass  # The image is already stored in current_body_image via update_current_state

    def update_mean_encoding(self) -> None:
        """Update the mean face encoding."""
        if self.face_encodings:
            self.mean_encoding = np.mean(self.face_encodings, axis=0)

    def set_name(self, name: str) -> None:
        """Set the name for this person."""
        self.name = name
        self.save_data()  # Save immediately when name is set

    def save_data(self) -> None:
        """Save person to disk."""
        try:
            from ..config import SAVE_DIR
        except ImportError:
            SAVE_DIR = "tracked_people"
            
        person_dir = os.path.join(SAVE_DIR, f"person_{self.track_id}")
        os.makedirs(person_dir, exist_ok=True)

        data = {
            "track_id": self.track_id,
            "name": self.name,
            "num_faces": len(self.face_images)
        }
        with open(os.path.join(person_dir, "data.json"), "w") as f:
            json.dump(data, f, indent=2)

        if self.face_encodings:
            np.save(os.path.join(person_dir, "encodings.npy"), np.array(self.face_encodings))

        # Save face images that aren't already saved
        saved_faces = [f for f in os.listdir(person_dir) if f.startswith('face_') and f.endswith('.jpg')]
        last_face_idx = len(saved_faces)
        for i, face_img in enumerate(self.face_images[last_face_idx:]):
            cv2.imwrite(os.path.join(person_dir, f"face_{last_face_idx + i + 1}.jpg"), face_img)

    def load_data(self) -> None:
        """Load person data from disk."""
        try:
            from ..config import SAVE_DIR
        except ImportError:
            SAVE_DIR = "tracked_people"
            
        person_dir = os.path.join(SAVE_DIR, f"person_{self.track_id}")
        json_path = os.path.join(person_dir, "data.json")
        encodings_path = os.path.join(person_dir, "encodings.npy")
        
        if os.path.exists(json_path) and os.path.exists(encodings_path):
            with open(json_path, "r") as f:
                data = json.load(f)
                self.name = data.get("name", None)
            
            self.face_encodings = np.load(encodings_path).tolist()
            self.update_mean_encoding()

    def get_display_label(self, certainty: float) -> str:
        """Get display label for this person."""
        label = f"ID {self.track_id} ({certainty:.1f}%)"
        if self.name:
            label += f" - {self.name}"
        return label

    def get_new_person_label(self) -> str:
        """Get label for newly detected person."""
        return f"NEW ID {self.track_id}"
    
    def get_current_face_image(self) -> Optional[np.ndarray]:
        """Get current face image."""
        return self.current_face_image
    
    def get_current_body_image(self) -> Optional[np.ndarray]:
        """Get current body image."""
        return self.current_body_image
    
    def get_current_pose_landmarks(self) -> Optional[Any]:
        """Get current pose landmarks."""
        return self.current_pose_landmarks
    
    def get_current_face_bbox(self) -> Optional[Tuple]:
        """Get current face bounding box."""
        return self.current_face_bbox
    
    def get_current_body_bbox(self) -> Optional[Tuple]:
        """Get current body bounding box."""
        return self.current_body_bbox
    
    def get_current_coordinates(self) -> Dict[str, Optional[Tuple]]:
        """Get current coordinates for face and body."""
        return {
            "face_bbox": self.current_face_bbox,
            "body_bbox": self.current_body_bbox
        }
    
    def get_all_face_images(self) -> List[np.ndarray]:
        """Get all stored face images."""
        return self.face_images.copy()
    
    def get_face_count(self) -> int:
        """Get number of stored face images."""
        return len(self.face_images)
    
    def mark_not_visible(self) -> None:
        """Mark person as not visible in current frame."""
        self.is_visible = False
