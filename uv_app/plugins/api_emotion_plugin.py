# uv_app/plugins/api_emotion_plugin.py

import cv2
import numpy as np
import requests
import json
from typing import Dict, Any
from .base import FacePlugin
from ..core.logging import get_logger

logger = get_logger()


class APIEmotionPlugin(FacePlugin):
    """Plugin for detecting emotions from face images using an external API."""
    
    def __init__(self, api_url: str = "http://localhost:8080", update_interval_ms: int = 2000):
        super().__init__("api_emotion", update_interval_ms)
        self.api_url = api_url
        self.health_endpoint = f"{api_url}/health"
        self.detect_endpoint = f"{api_url}/detect"
        logger.info(f"Initialized APIEmotionPlugin with API URL: {api_url}")
        
        # Check if API is available
        self._check_api_health()
    
    def _check_api_health(self) -> bool:
        """Check if the emotion API is healthy and available."""
        try:
            response = requests.get(self.health_endpoint, timeout=5)
            if response.status_code == 200:
                logger.info("✅ Emotion API is healthy and available")
                return True
            else:
                logger.warning(f"⚠️ Emotion API health check failed with status {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            logger.warning(f"⚠️ Cannot connect to emotion API: {e}")
            return False
    
    def process_face(self, face_image: np.ndarray, person) -> Dict[str, Any]:
        """Process face image to detect emotion using external API."""
        try:
            # Encode image as JPEG
            _, img_encoded = cv2.imencode('.jpg', face_image)
            img_bytes = img_encoded.tobytes()
            
            # Send to emotion detection API
            files = {'file': ('face.jpg', img_bytes, 'image/jpeg')}
            response = requests.post(
                self.detect_endpoint,
                files=files,
                timeout=10  # 10 second timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract emotion data from API response
                if "faces" in result and len(result["faces"]) > 0:
                    face_data = result["faces"][0]
                    scores = face_data.get("scores", {})
                    top_emotion = face_data.get("top_emotion", "unknown")
                    
                    # Find the confidence for the top emotion
                    confidence = scores.get(top_emotion, 0.0)
                    
                    emotion_data = {
                        "emotion": top_emotion,
                        "confidence": confidence,
                        "all_scores": scores,
                        "method": "api_emotion_detection"
                    }
                    
                    # Return emotion data without logging (plugin manager handles logging)
                    return emotion_data
                else:
                    # No faces detected in the image
                    emotion_data = {
                        "emotion": "unknown",
                        "confidence": 0.0,
                        "method": "api_emotion_detection",
                        "note": "No faces detected by API"
                    }
                    logger.warning(f"No faces detected for person {person.track_id}")
                    return emotion_data
            else:
                error_msg = f"API request failed with status {response.status_code}"
                logger.error(error_msg)
                return {"error": error_msg, "method": "api_emotion_detection"}
                
        except requests.exceptions.RequestException as e:
            error_msg = f"Error connecting to emotion API: {e}"
            logger.error(error_msg)
            return {"error": error_msg, "method": "api_emotion_detection"}
        except json.JSONDecodeError as e:
            error_msg = f"Error parsing API response: {e}"
            logger.error(error_msg)
            return {"error": error_msg, "method": "api_emotion_detection"}
        except Exception as e:
            error_msg = f"Error processing emotion detection: {e}"
            logger.error(error_msg)
            return {"error": error_msg, "method": "api_emotion_detection"}