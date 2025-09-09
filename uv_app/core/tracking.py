# uv_app/core/tracking.py

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from .detection import FaceDetector, BodyDetector
from .recognition import FaceRecognizer
from .person import TrackedPerson
from config import RESIZE_MAX


class PersonTracker:
    """Main tracking system that coordinates detection, recognition, and tracking."""
    
    def __init__(self, enable_face: bool = True, enable_body: bool = True, enable_pose: bool = True):
        self.enable_face = enable_face
        self.enable_body = enable_body
        self.enable_pose = enable_pose
        
        # Initialize components
        self.face_detector = FaceDetector() if enable_face else None
        self.body_detector = BodyDetector(enable_body, enable_pose) if (enable_body or enable_pose) else None
        self.recognizer = FaceRecognizer() if enable_face else None
        
        # Initialize plugin system
        self.plugin_manager = None
        self._init_plugin_system()
        
        # Load existing people
        if self.recognizer:
            self.recognizer.load_existing_people()
    
    def _init_plugin_system(self):
        """Initialize the plugin system."""
        try:
            from ..plugins.manager import PluginManager
            self.plugin_manager = PluginManager()
            print("✅ Plugin system initialized")
        except ImportError as e:
            print(f"⚠️  Plugin system not available: {e}")
            self.plugin_manager = None
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame and return annotated frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Annotated frame with tracking information
        """
        # Resize frame if needed
        frame = self._resize_frame(frame)
        current_frame_ids = set()
        
        # Process face detection and recognition
        if self.enable_face and self.face_detector and self.recognizer:
            frame = self._process_faces(frame, current_frame_ids)
        
        # Process body detection and pose estimation
        if (self.enable_body or self.enable_pose) and self.body_detector:
            frame = self._process_bodies_and_poses(frame)
        
        # Update current state for all visible people
        self._update_people_current_state()
        
        # Process plugins
        if self.plugin_manager:
            visible_people = self.get_visible_people()
            self.plugin_manager.process_people(visible_people, frame)
        
        # Update tracking state
        if self.recognizer:
            self.recognizer.update_tracking(current_frame_ids)
        
        return frame
    
    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame if larger than max allowed size."""
        h, w = frame.shape[:2]
        if max(h, w) > RESIZE_MAX:
            scale = RESIZE_MAX / max(h, w)
            return cv2.resize(frame, (int(w * scale), int(h * scale)))
        return frame
    
    def _process_faces(self, frame: np.ndarray, current_frame_ids: set) -> np.ndarray:
        """Process face detection and recognition."""
        from ui.display import FaceDisplayManager
        
        # Detect faces
        face_locations, face_encodings = self.face_detector.detect_faces(frame)
        
        # Process each detected face
        for i, (top, right, bottom, left) in enumerate(face_locations):
            face_encoding = face_encodings[i]
            face_img = self.face_detector.get_face_roi(frame, (top, right, bottom, left))
            bbox = (top, right, bottom, left)
            
            # Try to match with existing person
            matched_person = self.recognizer.process_face(face_encoding, face_img, bbox, frame)
            
            if matched_person:
                current_frame_ids.add(matched_person.track_id)
                # Update current state
                matched_person.update_current_state(
                    face_image=face_img,
                    face_bbox=bbox
                )
                # Calculate certainty and draw label
                _, distance = self.recognizer.find_best_match(face_encoding)
                certainty = self.recognizer.get_certainty_percentage(distance)
                label = matched_person.get_display_label(certainty)
                FaceDisplayManager.draw_face_box(frame, bbox, label)
        
        # Process candidate faces
        new_people = self.recognizer.process_candidates(frame)
        for person in new_people:
            current_frame_ids.add(person.track_id)
            # Find the candidate that created this person
            for candidate in self.recognizer.candidate_faces:
                if candidate["count"] >= 1:  # This was just processed
                    label = person.get_new_person_label()
                    FaceDisplayManager.draw_face_box(frame, candidate["bbox"], label, color=(0, 0, 255))
                    break
        
        return frame
    
    def _process_bodies_and_poses(self, frame: np.ndarray) -> np.ndarray:
        """Process body detection and pose estimation."""
        annotated_frame, bodies, poses = self.body_detector.detect_bodies_and_poses(frame)
        
        # Assign body data to tracked people
        if self.recognizer and bodies:
            for person in self.recognizer.tracked_people:
                if bodies:
                    person.add_body_data(bodies[-1], poses[-1] if poses else None)
        
        return annotated_frame
    
    def _update_people_current_state(self) -> None:
        """Update current state for all tracked people."""
        if not self.recognizer:
            return
        
        # Mark all people as not visible first
        for person in self.recognizer.tracked_people:
            person.mark_not_visible()
        
        # Update visible people with current data
        for person in self.recognizer.tracked_people:
            if person.is_visible:
                # Update with current frame data if available
                pass  # This will be handled by the face/body processing
    
    def get_visible_people(self) -> List[TrackedPerson]:
        """Get list of currently visible people."""
        if not self.recognizer:
            return []
        return [p for p in self.recognizer.tracked_people if p.is_visible]
    
    def get_all_people(self) -> List[TrackedPerson]:
        """Get list of all tracked people (visible and lost)."""
        if not self.recognizer:
            return []
        return self.recognizer.get_all_people()
    
    def get_person_by_id(self, track_id: int) -> Optional[TrackedPerson]:
        """Get person by track ID."""
        if not self.recognizer:
            return None
        return self.recognizer.get_person_by_id(track_id)
    
    def register_plugin(self, plugin) -> None:
        """Register a plugin."""
        if self.plugin_manager:
            self.plugin_manager.register_plugin(plugin)
        else:
            print("Plugin system not available")
    
    def get_plugin_results(self, person_id: int = None, plugin_name: str = None) -> Dict:
        """Get plugin results."""
        if not self.plugin_manager:
            return {}
        
        if person_id is not None:
            return self.plugin_manager.get_results_for_person(person_id)
        elif plugin_name is not None:
            return self.plugin_manager.get_results_for_plugin(plugin_name)
        else:
            return self.plugin_manager.get_all_results()
    
    def save_all_data(self) -> None:
        """Save all tracked people data."""
        if self.recognizer:
            for person in self.recognizer.get_all_people():
                person.save_data()
            print(f"Saved data for {len(self.recognizer.tracked_people)} tracked people and {len(self.recognizer.lost_people)} lost people")
    
    def get_tracked_people(self) -> List[TrackedPerson]:
        """Get list of currently tracked people."""
        if self.recognizer:
            return self.recognizer.tracked_people
        return []
    
    def get_lost_people(self) -> List[TrackedPerson]:
        """Get list of lost people."""
        if self.recognizer:
            return self.recognizer.lost_people
        return []
    
    def get_person_by_id(self, track_id: int) -> Optional[TrackedPerson]:
        """Get person by track ID."""
        if self.recognizer:
            for person in self.recognizer.get_all_people():
                if person.track_id == track_id:
                    return person
        return None
