# uv_app/plugins/activity_plugin.py

import cv2
import numpy as np
import requests
import json
from typing import Dict, Any
from .base import BodyPlugin, PosePlugin


class ActivityPlugin(BodyPlugin):
    """Plugin for detecting activities from body images using external API."""
    
    def __init__(self, api_url: str = None, update_interval_ms: int = 3000):
        super().__init__("activity", update_interval_ms)
        self.api_url = api_url or "https://api.example.com/activity"
        self.mock_mode = api_url is None
    
    def process_body(self, body_image: np.ndarray, person) -> Dict[str, Any]:
        """Process body image to detect activity."""
        if self.mock_mode:
            return self._mock_activity_detection(body_image)
        
        try:
            return self._api_activity_detection(body_image)
        except Exception as e:
            print(f"Error in activity detection: {e}")
            return self._mock_activity_detection(body_image)
    
    def _api_activity_detection(self, body_image: np.ndarray) -> Dict[str, Any]:
        """Detect activity using external API."""
        # Encode image as base64
        _, buffer = cv2.imencode('.jpg', body_image)
        image_base64 = buffer.tobytes()
        
        # Prepare API request
        headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer YOUR_API_KEY'  # Replace with actual API key
        }
        
        payload = {
            'image': image_base64.decode('utf-8'),
            'model': 'activity_detection'
        }
        
        # Make API request
        response = requests.post(
            self.api_url,
            headers=headers,
            json=payload,
            timeout=5
        )
        
        if response.status_code == 200:
            result = response.json()
            return {
                "activity": result.get('activity', 'unknown'),
                "confidence": result.get('confidence', 0.0),
                "method": "api"
            }
        else:
            raise Exception(f"API request failed: {response.status_code}")
    
    def _mock_activity_detection(self, body_image: np.ndarray) -> Dict[str, Any]:
        """Mock activity detection when API is not available."""
        # Analyze body image properties
        gray = cv2.cvtColor(body_image, cv2.COLOR_BGR2GRAY)
        
        # Calculate image statistics
        brightness = np.mean(gray)
        contrast = np.std(gray)
        height, width = body_image.shape[:2]
        aspect_ratio = height / width
        
        # Simple heuristics for activity detection
        if aspect_ratio > 2.0:
            activity = "standing"
            confidence = 0.7
        elif aspect_ratio < 1.2:
            activity = "sitting"
            confidence = 0.6
        elif contrast > 50:
            activity = "moving"
            confidence = 0.5
        else:
            activity = "stationary"
            confidence = 0.8
        
        return {
            "activity": activity,
            "confidence": confidence,
            "aspect_ratio": float(aspect_ratio),
            "brightness": float(brightness),
            "contrast": float(contrast),
            "method": "mock"
        }


class PoseActivityPlugin(PosePlugin):
    """Plugin for detecting activities from pose landmarks."""
    
    def __init__(self, update_interval_ms: int = 1000):
        super().__init__("pose_activity", update_interval_ms)
    
    def process_pose(self, pose_landmarks: Any, person) -> Dict[str, Any]:
        """Process pose landmarks to detect activity."""
        try:
            # Extract key points
            landmarks = pose_landmarks.landmark
            
            # Get key body points
            nose = landmarks[0]
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_elbow = landmarks[13]
            right_elbow = landmarks[14]
            left_wrist = landmarks[15]
            right_wrist = landmarks[16]
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            left_knee = landmarks[25]
            right_knee = landmarks[26]
            
            # Calculate angles and positions
            shoulder_angle = self._calculate_angle(left_shoulder, right_shoulder)
            left_arm_angle = self._calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_arm_angle = self._calculate_angle(right_shoulder, right_elbow, right_wrist)
            
            # Detect activities based on pose
            activity = self._detect_activity_from_pose(
                nose, left_shoulder, right_shoulder,
                left_arm_angle, right_arm_angle
            )
            
            return {
                "activity": activity,
                "shoulder_angle": float(shoulder_angle),
                "left_arm_angle": float(left_arm_angle),
                "right_arm_angle": float(right_arm_angle),
                "method": "pose_analysis"
            }
            
        except Exception as e:
            return {
                "activity": "unknown",
                "error": str(e),
                "method": "pose_analysis"
            }
    
    def _calculate_angle(self, point1, point2, point3=None):
        """Calculate angle between points."""
        if point3 is None:
            # Calculate angle from horizontal
            return np.arctan2(point2.y - point1.y, point2.x - point1.x) * 180 / np.pi
        
        # Calculate angle between three points
        a = np.array([point1.x, point1.y])
        b = np.array([point2.x, point2.y])
        c = np.array([point3.x, point3.y])
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return angle * 180 / np.pi
    
    def _detect_activity_from_pose(self, nose, left_shoulder, right_shoulder, 
                                  left_arm_angle, right_arm_angle):
        """Detect activity based on pose analysis."""
        # Check if person is looking down (phone usage)
        if nose.y > (left_shoulder.y + right_shoulder.y) / 2 + 0.1:
            return "using_phone"
        
        # Check for waving
        if abs(left_arm_angle) > 45 or abs(right_arm_angle) > 45:
            return "waving"
        
        # Check for hands up
        if left_arm_angle < -30 or right_arm_angle < -30:
            return "hands_up"
        
        # Check for crossed arms
        if abs(left_arm_angle) < 20 and abs(right_arm_angle) < 20:
            return "crossed_arms"
        
        return "neutral"


class SimpleActivityPlugin(BodyPlugin):
    """Simple activity plugin using basic image analysis."""
    
    def __init__(self, update_interval_ms: int = 2000):
        super().__init__("simple_activity", update_interval_ms)
    
    def process_body(self, body_image: np.ndarray, person) -> Dict[str, Any]:
        """Process body image using simple analysis."""
        # Convert to grayscale
        gray = cv2.cvtColor(body_image, cv2.COLOR_BGR2GRAY)
        
        # Calculate basic statistics
        brightness = np.mean(gray)
        contrast = np.std(gray)
        height, width = body_image.shape[:2]
        aspect_ratio = height / width
        
        # Detect edges for movement analysis
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (height * width)
        
        # Simple activity classification
        if edge_density > 0.1:
            activity = "active"
            confidence = 0.6
        elif aspect_ratio > 1.8:
            activity = "standing"
            confidence = 0.7
        elif aspect_ratio < 1.3:
            activity = "sitting"
            confidence = 0.6
        else:
            activity = "stationary"
            confidence = 0.5
        
        return {
            "activity": activity,
            "confidence": confidence,
            "edge_density": float(edge_density),
            "aspect_ratio": float(aspect_ratio),
            "brightness": float(brightness),
            "contrast": float(contrast),
            "method": "simple_analysis"
        }
