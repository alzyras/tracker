# uv_app/plugins/api_plugin.py

import cv2
import numpy as np
import requests
import json
import base64
import time
from typing import Dict, Any
from .base import FacePlugin, BodyPlugin


class GenericAPIPlugin(FacePlugin):
    """Generic plugin for making API calls with face images."""
    
    def __init__(self, api_url: str, api_key: str = None, 
                 update_interval_ms: int = 5000, plugin_name: str = "api_face"):
        super().__init__(plugin_name, update_interval_ms)
        self.api_url = api_url
        self.api_key = api_key
        self.headers = {
            'Content-Type': 'application/json'
        }
        if api_key:
            self.headers['Authorization'] = f'Bearer {api_key}'
    
    def process_face(self, face_image: np.ndarray, person) -> Dict[str, Any]:
        """Process face image by sending to API."""
        try:
            # Encode image as base64
            _, buffer = cv2.imencode('.jpg', face_image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Prepare payload
            payload = {
                'image': image_base64,
                'person_id': person.track_id,
                'timestamp': int(time.time() * 1000)
            }
            
            # Make API request
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "api_result": result,
                    "status": "success",
                    "method": "api_call"
                }
            else:
                return {
                    "error": f"API request failed: {response.status_code}",
                    "response": response.text,
                    "method": "api_call"
                }
                
        except Exception as e:
            return {
                "error": str(e),
                "method": "api_call"
            }


class BodyAPIPlugin(BodyPlugin):
    """Generic plugin for making API calls with body images."""
    
    def __init__(self, api_url: str, api_key: str = None, 
                 update_interval_ms: int = 5000, plugin_name: str = "api_body"):
        super().__init__(plugin_name, update_interval_ms)
        self.api_url = api_url
        self.api_key = api_key
        self.headers = {
            'Content-Type': 'application/json'
        }
        if api_key:
            self.headers['Authorization'] = f'Bearer {api_key}'
    
    def process_body(self, body_image: np.ndarray, person) -> Dict[str, Any]:
        """Process body image by sending to API."""
        try:
            # Encode image as base64
            _, buffer = cv2.imencode('.jpg', body_image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Prepare payload
            payload = {
                'image': image_base64,
                'person_id': person.track_id,
                'timestamp': int(time.time() * 1000)
            }
            
            # Make API request
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "api_result": result,
                    "status": "success",
                    "method": "api_call"
                }
            else:
                return {
                    "error": f"API request failed: {response.status_code}",
                    "response": response.text,
                    "method": "api_call"
                }
                
        except Exception as e:
            return {
                "error": str(e),
                "method": "api_call"
            }


# Example usage functions
def create_emotion_api_plugin(api_url: str, api_key: str = None):
    """Create an emotion detection plugin using external API."""
    return GenericAPIPlugin(
        api_url=api_url,
        api_key=api_key,
        update_interval_ms=2000,
        plugin_name="emotion_api"
    )


def create_activity_api_plugin(api_url: str, api_key: str = None):
    """Create an activity detection plugin using external API."""
    return BodyAPIPlugin(
        api_url=api_url,
        api_key=api_key,
        update_interval_ms=3000,
        plugin_name="activity_api"
    )


def create_custom_api_plugin(api_url: str, api_key: str = None, 
                           plugin_name: str = "custom_api",
                           update_interval_ms: int = 5000):
    """Create a custom API plugin."""
    return GenericAPIPlugin(
        api_url=api_url,
        api_key=api_key,
        update_interval_ms=update_interval_ms,
        plugin_name=plugin_name
    )
