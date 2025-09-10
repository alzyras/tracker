# uv_app/core/tracking.py

import cv2
import numpy as np
import time
from typing import List, Tuple, Optional, Dict, Any
from .detection import FaceDetector, BodyDetector
from .recognition import FaceRecognizer
from .person import TrackedPerson
from .logging import get_logger
from config import RESIZE_MAX, PLUGIN_CONFIG

logger = get_logger()


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
        
        logger.info("Initialized PersonTracker")
    
    def _init_plugin_system(self):
        """Initialize the plugin system."""
        try:
            # Try relative import first
            try:
                from ..plugins.manager import PluginManager
                from ..plugins import PLUGIN_REGISTRY
            except (ImportError, ValueError):
                # Fallback to absolute import
                from uv_app.plugins.manager import PluginManager
                from uv_app.plugins import PLUGIN_REGISTRY
            self.plugin_manager = PluginManager()
            logger.info("✅ Plugin system initialized")
            
            # Register the API emotion plugin if enabled
            try:
                if PLUGIN_CONFIG.get('api_emotion_plugin_enabled', False):
                    api_emotion_cls = PLUGIN_REGISTRY.get("api_emotion")
                    if api_emotion_cls:
                        api_url = PLUGIN_CONFIG.get('api_emotion_api_url', 'http://localhost:8080')
                        interval = PLUGIN_CONFIG.get('api_emotion_plugin_interval', 2000)
                        api_emotion_plugin = api_emotion_cls(api_url=api_url, update_interval_ms=interval)
                        self.plugin_manager.register_plugin(api_emotion_plugin)
                        logger.info("✅ API emotion plugin registered")
            except Exception as e:
                logger.warning(f"⚠️  Could not register API emotion plugin: {e}")
            
            # Register the emotion logger plugin by default
            try:
                emotion_logger_cls = PLUGIN_REGISTRY.get("emotion_logger")
                if emotion_logger_cls:
                    emotion_logger = emotion_logger_cls()
                    self.plugin_manager.register_plugin(emotion_logger)
                    logger.debug("✅ Emotion logger plugin registered")
            except Exception as e:
                logger.warning(f"⚠️  Could not register emotion logger plugin: {e}")
            
            # Register the person event logger plugin by default
            try:
                person_event_logger_cls = PLUGIN_REGISTRY.get("person_event_logger")
                if person_event_logger_cls:
                    person_event_logger = person_event_logger_cls()
                    self.plugin_manager.register_plugin(person_event_logger)
                    logger.debug("✅ Person event logger plugin registered")
            except Exception as e:
                logger.warning(f"⚠️  Could not register person event logger plugin: {e}")
        except ImportError as e:
            logger.warning(f"⚠️  Plugin system not available: {e}")
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
        
        # Reset visibility for all people at the start of the frame
        if self.recognizer:
            for person in self.recognizer.tracked_people:
                person.mark_not_visible()

        # Process face detection and recognition
        if self.enable_face and self.face_detector and self.recognizer:
            frame = self._process_faces(frame, current_frame_ids)
        
        # Process body detection and pose estimation
        if (self.enable_body or self.enable_pose) and self.body_detector:
            frame = self._process_bodies_and_poses(frame)
        
        # Current state already updated in face/body processing
        
        # Process plugins
        if self.plugin_manager:
            visible_people = self.get_visible_people()
            self.plugin_manager.process_people(visible_people, frame)
            
            # Log emotions every 5 seconds
            self._log_emotions(visible_people)
        
        # Update tracking state
        if self.recognizer:
            self.recognizer.update_tracking(current_frame_ids, self.plugin_manager)
        
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
                
                # Get emotion from plugin results if available
                emotion = None
                emotion_conf = None
                if self.plugin_manager:
                    results = self.plugin_manager.get_results_for_person(matched_person.track_id)
                    # Check for API emotion results first, then fall back to other emotion plugins
                    emotion_result = results.get("api_emotion") or results.get("emotion") or results.get("simple_emotion")
                    if emotion_result and "emotion" in emotion_result:
                        emotion = emotion_result.get("emotion")
                        conf = emotion_result.get("confidence")
                        if isinstance(conf, dict) and isinstance(emotion, str):
                            emotion_conf = conf.get(emotion, None)
                        else:
                            emotion_conf = conf
                
                # Draw face box with emotion if available
                if emotion:
                    FaceDisplayManager.draw_face_box_with_emotion(frame, bbox, label, emotion, emotion_conf)
                else:
                    FaceDisplayManager.draw_face_box(frame, bbox, label)
                
                # Log the match
                logger.log_person_match(matched_person.name or f"ID {matched_person.track_id}", distance)
        
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
        
        # No-op: visibility is managed at frame start and during processing
        return
    
    def _log_emotions(self, people: List[TrackedPerson]) -> None:
        """Log emotion information for visible people every 5 seconds."""
        if not self.plugin_manager:
            return
        
        # Find the emotion logger plugin
        emotion_logger = None
        for plugin in self.plugin_manager.plugins:
            if plugin.name == "emotion_logger":
                emotion_logger = plugin
                break
        
        if emotion_logger:
            # Collect emotion data for all visible people
            current_time_ms = int(time.time() * 1000)
            
            # Only log every 5 seconds
            if hasattr(emotion_logger, 'last_log_time'):
                last_log = emotion_logger.last_log_time
            else:
                last_log = 0
                
            if current_time_ms - last_log >= emotion_logger.update_interval_ms:
                emotion_messages = []
                for person in people:
                    # Get emotion from plugin results
                    results = self.plugin_manager.get_results_for_person(person.track_id)
                    # Check for API emotion results first, then fall back to other emotion plugins
                    emotion_result = results.get("api_emotion") or results.get("emotion") or results.get("simple_emotion")
                    
                    if emotion_result and "emotion" in emotion_result:
                        emotion = emotion_result["emotion"]
                        confidence = emotion_result.get("confidence", 0.0)
                        
                        # If confidence is a dict (from API), get the value for the top emotion
                        if isinstance(confidence, dict):
                            confidence = confidence.get(emotion, 0.0)
                        
                        person_name = person.name if person.name else f"Person ID {person.track_id}"
                        emotion_messages.append(f"{person_name} is {emotion} ({confidence:.2f})")
                
                if emotion_messages:
                    logger.info(f"😊 Emotions: {', '.join(emotion_messages)}")
                
                # Update the logger timestamp
                emotion_logger.last_log_time = current_time_ms
    
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
            logger.warning("Plugin system not available")
    
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
            logger.info(f"Saved data for {len(self.recognizer.tracked_people)} tracked people and {len(self.recognizer.lost_people)} lost people")
    
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
