import json
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import MAX_FACE_IMAGES, SAVE_DIR


class TrackedPerson:
    """Represents a person being tracked with face encodings and metadata."""

    def __init__(self, track_id: int) -> None:
        self.track_id = track_id
        self.face_encodings: List[np.ndarray] = []
        self.face_images: List[np.ndarray] = []
        self.face_boxes: List[Tuple[int, int, int, int]] = []  # (top, right, bottom, left)
        self.body_boxes: List[Tuple[int, int, int, int]] = []  # (x1, y1, x2, y2)
        self.pose_landmarks: List = []  # list of pose objects
        self.missed_frames = 0
        self.name: Optional[str] = None
        self.mean_encoding: Optional[np.ndarray] = None
        self.load_data()

    def save_snapshot(self, frame: np.ndarray, event_name: str) -> None:
        """Save snapshot to the person's folder with event metadata."""
        person_dir = os.path.join(SAVE_DIR, f"person_{self.track_id}")
        os.makedirs(person_dir, exist_ok=True)
        filename = f"{event_name}_{len(os.listdir(person_dir))}.jpg"
        cv2.imwrite(os.path.join(person_dir, filename), frame)

    def update(self) -> None:
        """Reset missed frames counter."""
        self.missed_frames = 0

    def add_face_data(
        self,
        encoding: np.ndarray,
        face_img: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> None:
        """Add a new face encoding/image and optional face bbox."""
        if len(self.face_encodings) >= MAX_FACE_IMAGES:
            return

        # Avoid duplicate encodings
        if all(np.linalg.norm(encoding - f) > 0.3 for f in self.face_encodings):
            self.face_encodings.append(encoding)
            self.face_images.append(face_img)
            if bbox:
                self.face_boxes.append(bbox)
            self.update_mean_encoding()

    def add_body_data(
        self,
        body_bbox: Optional[Tuple[int, int, int, int]] = None,
        pose: Optional = None,
    ) -> None:
        """Store body bbox and pose data."""
        if body_bbox:
            self.body_boxes.append(body_bbox)
        if pose:
            self.pose_landmarks.append(pose)

    def update_mean_encoding(self) -> None:
        """Update the mean face encoding from all stored encodings."""
        if self.face_encodings:
            self.mean_encoding = np.mean(self.face_encodings, axis=0)

    def save_data(self) -> None:
        """Save person data to disk."""
        person_dir = os.path.join(SAVE_DIR, f"person_{self.track_id}")
        os.makedirs(person_dir, exist_ok=True)

        data = {
            "track_id": self.track_id,
            "name": self.name,
            "num_faces": len(self.face_images),
        }
        with open(os.path.join(person_dir, "data.json"), "w") as f:
            json.dump(data, f, indent=2)

        if self.face_encodings:
            np.save(os.path.join(person_dir, "encodings.npy"), np.array(self.face_encodings))

        saved_faces = [f for f in os.listdir(person_dir) if f.startswith("face_") and f.endswith(".jpg")]
        last_face_idx = len(saved_faces)
        for i, face_img in enumerate(self.face_images[last_face_idx:]):
            cv2.imwrite(os.path.join(person_dir, f"face_{last_face_idx + i + 1}.jpg"), face_img)

    def load_data(self) -> None:
        """Load person data from disk."""
        person_dir = os.path.join(SAVE_DIR, f"person_{self.track_id}")
        json_path = os.path.join(person_dir, "data.json")
        encodings_path = os.path.join(person_dir, "encodings.npy")
        if os.path.exists(json_path) and os.path.exists(encodings_path):
            with open(json_path) as f:
                data = json.load(f)
                self.name = data.get("name")
            self.face_encodings = np.load(encodings_path).tolist()
            self.update_mean_encoding()
