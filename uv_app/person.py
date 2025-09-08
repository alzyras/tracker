import os
import json
import numpy as np
import cv2
from config import SAVE_DIR, MAX_FACE_IMAGES

class TrackedPerson:
    """
    Represents a person being tracked.
    Stores face encodings, face/body bounding boxes, pose, and snapshots.
    """

    def __init__(self, track_id):
        self.track_id = track_id
        self.face_encodings = []
        self.face_images = []
        self.face_boxes = []       # (top, right, bottom, left)
        self.body_boxes = []       # (x1, y1, x2, y2)
        self.pose_landmarks = []   # list of pose objects
        self.current_emotion = None
        self.phone_detected = False
        self.emotion_history = []
        self.phone_history = []
        self.current_emotion = None
        self.phone_detected = False
        self.body_actions = {}
        self.snapshot_enabled = True  # Save snapshots of events
        self.history = []
        self.missed_frames = 0
        self.name = None
        self.mean_encoding = None
        self.load_data()

    def update_emotion(self, emotion, frame=None):
        if emotion and emotion != self.current_emotion:
            self.current_emotion = emotion
            print(f"Person #{self.track_id} is {emotion}")
            if self.snapshot_enabled and frame is not None:
                self.save_snapshot(frame, f"emotion_{emotion}")

    def update_phone_status(self, phone_detected, frame=None):
        if phone_detected != self.phone_detected:
            self.phone_detected = phone_detected
            if phone_detected:
                print(f"Person #{self.track_id} is looking at their phone")
                if self.snapshot_enabled and frame is not None:
                    self.save_snapshot(frame, "phone")

    def update_body_actions(self, actions, frame=None):
        for action, detected in actions.items():
            if detected and self.body_actions.get(action) != True:
                self.body_actions[action] = True
                print(f"Person #{self.track_id} is {action}")
                if self.snapshot_enabled and frame is not None:
                    self.save_snapshot(frame, f"action_{action}")

    def save_snapshot(self, frame, event_name):
        """Save snapshot to the person's folder with event metadata"""
        person_dir = os.path.join(SAVE_DIR, f"person_{self.track_id}")
        os.makedirs(person_dir, exist_ok=True)
        filename = f"{event_name}_{len(os.listdir(person_dir))}.jpg"
        cv2.imwrite(os.path.join(person_dir, filename), frame)

    def update(self):
        """Resets missed frames counter."""
        self.missed_frames = 0

    def add_face_data(self, encoding, face_img, bbox=None):
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

    def add_body_data(self, body_bbox=None, pose=None, body_img=None):
        """Store body bbox, pose, and optional body image."""
        if body_bbox:
            self.body_boxes.append(body_bbox)
        if pose:
            self.pose_landmarks.append(pose)
        # You could also store body_img in memory if needed

    def update_mean_encoding(self):
        if self.face_encodings:
            self.mean_encoding = np.mean(self.face_encodings, axis=0)

    def save_data(self):
        """Save person to disk."""
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

        saved_faces = [f for f in os.listdir(person_dir) if f.startswith('face_') and f.endswith('.jpg')]
        last_face_idx = len(saved_faces)
        for i, face_img in enumerate(self.face_images[last_face_idx:]):
            cv2.imwrite(os.path.join(person_dir, f"face_{last_face_idx + i + 1}.jpg"), face_img)

    def load_data(self):
        """Load person data from disk."""
        person_dir = os.path.join(SAVE_DIR, f"person_{self.track_id}")
        json_path = os.path.join(person_dir, "data.json")
        encodings_path = os.path.join(person_dir, "encodings.npy")
        if os.path.exists(json_path) and os.path.exists(encodings_path):
            with open(json_path, "r") as f:
                data = json.load(f)
                self.name = data.get("name", None)
            self.face_encodings = np.load(encodings_path).tolist()
            self.update_mean_encoding()
