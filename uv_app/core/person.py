# uv_app/core/person.py

import os
import json
import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict, Any
from config import SAVE_DIR, MAX_FACE_IMAGES


class TrackedPerson:
    """
    Represents a person being tracked.
    Stores face encodings, face/body bounding boxes, pose, and snapshots.
    """

    def __init__(self, track_id: int):
        self.track_id = track_id
        self.face_encodings: List[np.ndarray] = []
        self.face_images: List[np.ndarray] = []
        self.face_boxes: List[Tuple] = []  # (top, right, bottom, left)
        self.body_boxes: List[Tuple] = []  # (x1, y1, x2, y2)
        self.pose_landmarks: List[Any] = []  # list of pose objects
        self.current_emotion: Optional[str] = None
        self.phone_detected: bool = False
        self.emotion_history: List[str] = []
        self.phone_history: List[bool] = []
        self.body_actions: Dict[str, bool] = {}
        self.snapshot_enabled: bool = True  # Save snapshots of events
        self.history: List[Dict] = []
        self.missed_frames: int = 0
        self.name: Optional[str] = None
        self.mean_encoding: Optional[np.ndarray] = None
        self.load_data()

    def update_emotion(self, emotion: str, frame: Optional[np.ndarray] = None) -> None:
        """Update person's current emotion."""
        if emotion and emotion != self.current_emotion:
            self.current_emotion = emotion
            self.emotion_history.append(emotion)
            print(f"Person #{self.track_id} is {emotion}")
            if self.snapshot_enabled and frame is not None:
                self.save_snapshot(frame, f"emotion_{emotion}")

    def update_phone_status(self, phone_detected: bool, frame: Optional[np.ndarray] = None) -> None:
        """Update person's phone detection status."""
        if phone_detected != self.phone_detected:
            self.phone_detected = phone_detected
            self.phone_history.append(phone_detected)
            if phone_detected:
                print(f"Person #{self.track_id} is looking at their phone")
                if self.snapshot_enabled and frame is not None:
                    self.save_snapshot(frame, "phone")

    def update_body_actions(self, actions: Dict[str, bool], frame: Optional[np.ndarray] = None) -> None:
        """Update person's body actions."""
        for action, detected in actions.items():
            if detected and self.body_actions.get(action) != True:
                self.body_actions[action] = True
                print(f"Person #{self.track_id} is {action}")
                if self.snapshot_enabled and frame is not None:
                    self.save_snapshot(frame, f"action_{action}")

    def save_snapshot(self, frame: np.ndarray, event_name: str) -> None:
        """Save snapshot to the person's folder with event metadata."""
        person_dir = os.path.join(SAVE_DIR, f"person_{self.track_id}")
        os.makedirs(person_dir, exist_ok=True)
        filename = f"{event_name}_{len(os.listdir(person_dir))}.jpg"
        cv2.imwrite(os.path.join(person_dir, filename), frame)

    def save_face_image(self, face_img: np.ndarray) -> None:
        """Save individual face image to person's folder."""
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
        if len(self.face_encodings) >= MAX_FACE_IMAGES:
            return

        # Avoid duplicate encodings (use a more lenient threshold)
        if all(np.linalg.norm(encoding - f) > 0.4 for f in self.face_encodings):
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
        # You could also store body_img in memory if needed

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
        person_dir = os.path.join(SAVE_DIR, f"person_{self.track_id}")
        os.makedirs(person_dir, exist_ok=True)

        data = {
            "track_id": self.track_id,
            "name": self.name,
            "num_faces": len(self.face_images),
            "emotion_history": self.emotion_history,
            "phone_history": self.phone_history,
            "body_actions": self.body_actions
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
        person_dir = os.path.join(SAVE_DIR, f"person_{self.track_id}")
        json_path = os.path.join(person_dir, "data.json")
        encodings_path = os.path.join(person_dir, "encodings.npy")
        
        if os.path.exists(json_path) and os.path.exists(encodings_path):
            with open(json_path, "r") as f:
                data = json.load(f)
                self.name = data.get("name", None)
                self.emotion_history = data.get("emotion_history", [])
                self.phone_history = data.get("phone_history", [])
                self.body_actions = data.get("body_actions", {})
            
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
