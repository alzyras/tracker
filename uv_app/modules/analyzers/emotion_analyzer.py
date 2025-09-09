"""Emotion analyzer using FER (Facial Expression Recognition)."""

from typing import Any, Dict

import cv2
import numpy as np
from fer import FER

from .base import BaseAnalyzer
from ...person import TrackedPerson


class EmotionAnalyzer(BaseAnalyzer):
    """Analyzes facial emotions using FER."""
    
    def __init__(self) -> None:
        """Initialize the emotion analyzer."""
        super().__init__("emotion")
        self.detector = FER(mtcnn=True)
    
    def analyze(
        self, 
        person: TrackedPerson, 
        frame: np.ndarray,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Analyze emotions in the person's face.
        
        Args:
            person: The tracked person
            frame: Current frame
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing emotion analysis results
        """
        if not person.face_boxes:
            return {"emotions": {}, "dominant_emotion": None, "confidence": 0.0}
        
        # Get the latest face box
        top, right, bottom, left = person.face_boxes[-1]
        
        # Extract face region
        face_region = frame[top:bottom, left:right]
        
        if face_region.size == 0:
            return {"emotions": {}, "dominant_emotion": None, "confidence": 0.0}
        
        # Analyze emotions
        emotions = self.detector.detect_emotions(face_region)
        
        if not emotions:
            return {"emotions": {}, "dominant_emotion": None, "confidence": 0.0}
        
        # Get the first (and likely only) detection
        emotion_data = emotions[0]["emotions"]
        
        # Find dominant emotion
        dominant_emotion = max(emotion_data, key=emotion_data.get)
        confidence = emotion_data[dominant_emotion]
        
        return {
            "emotions": emotion_data,
            "dominant_emotion": dominant_emotion,
            "confidence": confidence,
        }
    
    def get_config(self) -> Dict[str, Any]:
        """Get analyzer configuration."""
        return {
            "name": self.name,
            "type": "emotion",
            "backend": "FER",
            "uses_face": True,
            "uses_body": False,
            "uses_pose": False,
        }
    
    def should_analyze(self, person: TrackedPerson) -> bool:
        """Check if person has face data for emotion analysis."""
        return bool(person.face_boxes)