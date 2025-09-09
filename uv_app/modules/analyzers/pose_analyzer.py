"""Pose analyzer for detecting body poses and actions."""

from typing import Any, Dict

import numpy as np

from .base import BaseAnalyzer
from ...person import TrackedPerson


class PoseAnalyzer(BaseAnalyzer):
    """Analyzes body poses and detects common actions."""
    
    def __init__(self) -> None:
        """Initialize the pose analyzer."""
        super().__init__("pose")
        
        # Define pose landmark indices (MediaPipe format)
        self.POSE_LANDMARKS = {
            "nose": 0,
            "left_eye_inner": 1,
            "left_eye": 2,
            "left_eye_outer": 3,
            "right_eye_inner": 4,
            "right_eye": 5,
            "right_eye_outer": 6,
            "left_ear": 7,
            "right_ear": 8,
            "mouth_left": 9,
            "mouth_right": 10,
            "left_shoulder": 11,
            "right_shoulder": 12,
            "left_elbow": 13,
            "right_elbow": 14,
            "left_wrist": 15,
            "right_wrist": 16,
            "left_pinky": 17,
            "right_pinky": 18,
            "left_index": 19,
            "right_index": 20,
            "left_thumb": 21,
            "right_thumb": 22,
            "left_hip": 23,
            "right_hip": 24,
            "left_knee": 25,
            "right_knee": 26,
            "left_ankle": 27,
            "right_ankle": 28,
            "left_heel": 29,
            "right_heel": 30,
            "left_foot_index": 31,
            "right_foot_index": 32,
        }
    
    def analyze(
        self, 
        person: TrackedPerson, 
        frame: np.ndarray,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Analyze pose and detect actions.
        
        Args:
            person: The tracked person
            frame: Current frame
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing pose analysis results
        """
        if not person.pose_landmarks:
            return {
                "pose_detected": False,
                "actions": {},
                "pose_quality": 0.0,
            }
        
        # Get the latest pose
        pose = person.pose_landmarks[-1]
        landmarks = pose.landmark
        
        # Calculate pose quality (average visibility)
        pose_quality = sum(lm.visibility for lm in landmarks) / len(landmarks)
        
        # Detect actions
        actions = self._detect_actions(landmarks)
        
        return {
            "pose_detected": True,
            "actions": actions,
            "pose_quality": pose_quality,
            "landmark_count": len(landmarks),
        }
    
    def _detect_actions(self, landmarks) -> Dict[str, bool]:
        """Detect common actions from pose landmarks.
        
        Args:
            landmarks: MediaPipe pose landmarks
            
        Returns:
            Dictionary of detected actions
        """
        actions = {}
        
        try:
            # Get key landmarks
            left_wrist = landmarks[self.POSE_LANDMARKS["left_wrist"]]
            right_wrist = landmarks[self.POSE_LANDMARKS["right_wrist"]]
            left_shoulder = landmarks[self.POSE_LANDMARKS["left_shoulder"]]
            right_shoulder = landmarks[self.POSE_LANDMARKS["right_shoulder"]]
            nose = landmarks[self.POSE_LANDMARKS["nose"]]
            
            # Hands raised detection
            hands_raised = (
                left_wrist.y < left_shoulder.y and 
                right_wrist.y < right_shoulder.y
            )
            actions["hands_raised"] = hands_raised
            
            # Waving detection (simplified - one hand raised)
            waving = (
                left_wrist.y < left_shoulder.y or 
                right_wrist.y < right_shoulder.y
            )
            actions["waving"] = waving
            
            # Phone usage detection (hand near face)
            phone_distance_threshold = 0.1
            left_phone = (
                abs(left_wrist.x - nose.x) < phone_distance_threshold and
                abs(left_wrist.y - nose.y) < phone_distance_threshold
            )
            right_phone = (
                abs(right_wrist.x - nose.x) < phone_distance_threshold and
                abs(right_wrist.y - nose.y) < phone_distance_threshold
            )
            actions["using_phone"] = left_phone or right_phone
            
        except (IndexError, AttributeError):
            # Handle missing landmarks gracefully
            actions = {
                "hands_raised": False,
                "waving": False,
                "using_phone": False,
            }
        
        return actions
    
    def get_config(self) -> Dict[str, Any]:
        """Get analyzer configuration."""
        return {
            "name": self.name,
            "type": "pose",
            "backend": "MediaPipe",
            "uses_face": False,
            "uses_body": False,
            "uses_pose": True,
            "actions": ["hands_raised", "waving", "using_phone"],
        }
    
    def should_analyze(self, person: TrackedPerson) -> bool:
        """Check if person has pose data for analysis."""
        return bool(person.pose_landmarks)