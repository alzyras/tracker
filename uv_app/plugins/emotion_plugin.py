# uv_app/plugins/emotion_plugin.py

import cv2
import numpy as np
from typing import Dict, Any
from .base import FacePlugin

# Fix the import issue by using absolute import
try:
    from ..core.logging import get_logger
except (ImportError, ValueError):
    # Fallback to absolute import
    from uv_app.core.logging import get_logger

logger = get_logger()


class EmotionPlugin(FacePlugin):
    """Plugin for detecting emotions from face images."""
    
    def __init__(self, update_interval_ms: int = 2000):
        super().__init__("emotion", update_interval_ms)
        self.emotion_model = None
        self._load_model()
    
    def _load_model(self):
        """Load emotion detection model."""
        try:
            # Try to import deepface for emotion detection
            import deepface
            from deepface import DeepFace
            self.emotion_model = DeepFace
            logger.info("✅ Emotion plugin loaded with DeepFace")
        except ImportError:
            logger.warning("⚠️ DeepFace not available. Using mock emotion detection.")
            self.emotion_model = None
    
    def process_face(self, face_image: np.ndarray, person) -> Dict[str, Any]:
        """Process face image to detect emotion."""
        if self.emotion_model is None:
            return self._mock_emotion_detection(face_image)
        
        try:
            # Use DeepFace for emotion detection
            result = self.emotion_model.analyze(
                face_image, 
                actions=['emotion'], 
                enforce_detection=False
            )
            
            if isinstance(result, list):
                result = result[0]
            
            emotion = result.get('dominant_emotion', 'unknown')
            confidence = result.get('emotion', {})
            
            return {
                "emotion": emotion,
                "confidence": confidence,
                "method": "deepface"
            }
            
        except Exception as e:
            logger.error(f"Error in emotion detection: {e}")
            return self._mock_emotion_detection(face_image)
    
    def _mock_emotion_detection(self, face_image: np.ndarray) -> Dict[str, Any]:
        """Mock emotion detection when DeepFace is not available."""
        # Simple mock based on image properties
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        if brightness > 150:
            emotion = "happy"
            confidence = 0.7
        elif brightness < 100:
            emotion = "sad"
            confidence = 0.6
        else:
            emotion = "neutral"
            confidence = 0.8
        
        return {
            "emotion": emotion,
            "confidence": {emotion: confidence},
            "method": "mock"
        }


class SimpleEmotionPlugin(FacePlugin):
    """Simple emotion plugin using basic image analysis."""
    
    def __init__(self, update_interval_ms: int = 1000):
        super().__init__("simple_emotion", update_interval_ms)
    
    def process_face(self, face_image: np.ndarray, person) -> Dict[str, Any]:
        """Process face using simple image analysis."""
        # Convert to grayscale
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        # Analyze brightness and contrast
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        # Simple heuristics
        if brightness > 140 and contrast > 30:
            emotion = "happy"
            confidence = 0.6
        elif brightness < 110 and contrast < 25:
            emotion = "sad"
            confidence = 0.5
        elif contrast > 40:
            emotion = "surprised"
            confidence = 0.4
        else:
            emotion = "neutral"
            confidence = 0.7
        
        return {
            "emotion": emotion,
            "confidence": confidence,
            "brightness": float(brightness),
            "contrast": float(contrast),
            "method": "simple_analysis"
        }
