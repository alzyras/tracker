# uv_app/core/detection.py

import cv2
import face_recognition
import numpy as np
from typing import List, Tuple, Optional
from config import RESIZE_MAX
from .logging import get_logger

logger = get_logger()


class FaceDetector:
    """Handles face detection and encoding extraction."""
    
    def __init__(self):
        self.face_locations = []
        self.face_encodings = []
        logger.debug("Initialized FaceDetector")
    
    def detect_faces(self, frame: np.ndarray) -> Tuple[List[Tuple], List[np.ndarray]]:
        """
        Detect faces in a frame and return locations and encodings.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Tuple of (face_locations, face_encodings)
        """
        # Convert BGR to RGB for face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect face locations
        self.face_locations = face_recognition.face_locations(rgb_frame)
        logger.log_detection("faces", len(self.face_locations))
        
        # Extract face encodings
        self.face_encodings = face_recognition.face_encodings(rgb_frame, self.face_locations)
        
        return self.face_locations, self.face_encodings
    
    def get_face_roi(self, frame: np.ndarray, location: Tuple) -> np.ndarray:
        """
        Extract face region of interest from frame.
        
        Args:
            frame: Input frame
            location: Face location tuple (top, right, bottom, left)
            
        Returns:
            Face ROI as numpy array
        """
        top, right, bottom, left = location
        return frame[top:bottom, left:right]


class BodyDetector:
    """Handles body detection and pose estimation."""
    
    def __init__(self, enable_body: bool = True, enable_pose: bool = True):
        self.enable_body = enable_body
        self.enable_pose = enable_pose
        self._init_mediapipe()
        logger.debug(f"Initialized BodyDetector (body: {enable_body}, pose: {enable_pose})")
    
    def _init_mediapipe(self):
        """Initialize MediaPipe pose model."""
        try:
            import mediapipe as mp
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.pose_model = self.mp_pose.Pose(
                static_image_mode=False, 
                min_detection_confidence=0.5
            )
            logger.info("✅ MediaPipe pose model loaded successfully")
        except ImportError:
            logger.warning("⚠️ MediaPipe not available. Body detection disabled.")
            self.pose_model = None
    
    def detect_bodies_and_poses(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Tuple], List]:
        """
        Detect bodies and poses in frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (annotated_frame, body_boxes, poses)
        """
        if not self.pose_model:
            return frame, [], []
        
        bodies = []
        poses = []
        annotated = frame.copy()
        
        if self.enable_pose or self.enable_body:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose_model.process(rgb)
            
            if results.pose_landmarks:
                if self.enable_pose:
                    self.mp_drawing.draw_landmarks(
                        annotated,
                        results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                            color=(0, 255, 0), thickness=2, circle_radius=2
                        ),
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(
                            color=(0, 0, 255), thickness=2
                        )
                    )
                
                if self.enable_body:
                    h, w, _ = frame.shape
                    xs = [lm.x * w for lm in results.pose_landmarks.landmark]
                    ys = [lm.y * h for lm in results.pose_landmarks.landmark]
                    x1, y1 = int(min(xs)), int(min(ys))
                    x2, y2 = int(max(xs)), int(max(ys))
                    bodies.append((x1, y1, x2, y2))
                    
                    # Draw bounding box
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                poses.append(results.pose_landmarks)
        
        logger.log_detection("bodies", len(bodies))
        logger.log_detection("poses", len(poses))
        return annotated, bodies, poses
