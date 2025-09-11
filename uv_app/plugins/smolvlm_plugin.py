# uv_app/plugins/smolvlm_plugin.py

import cv2
import numpy as np
import requests
import json
import base64
import time
import threading
from typing import Dict, Any
from .base import BodyPlugin

# Fix the import issue by using absolute import
try:
    from ..core.logging import get_logger
except (ImportError, ValueError):
    # Fallback to absolute import
    from uv_app.core.logging import get_logger

logger = get_logger()


class SmolVLMPlugin(BodyPlugin):
    """Plugin for detecting activities using the SmolVLM API."""
    
    def __init__(self, api_url: str = "http://localhost:9000/describe", 
                 api_key: str = None, update_interval_ms: int = 5000):
        super().__init__("smolvlm_activity", update_interval_ms)
        self.api_url = api_url
        self.api_key = api_key
        self.headers = {}
        if api_key:
            self.headers['Authorization'] = f'Bearer {api_key}'
        
        # Store pending requests to avoid blocking
        self.pending_requests = {}
        
        logger.info(f"✅ SmolVLM plugin initialized with API URL: {api_url}")
    
    def process_body(self, body_image: np.ndarray, person) -> Dict[str, Any]:
        """Process body image by sending to SmolVLM API."""
        try:
            # Check if we already have a pending request for this person
            person_id = person.track_id
            if person_id in self.pending_requests:
                # Return previous result if still pending
                return self.pending_requests[person_id].get("last_result", {})
            
            # Make async API request in a separate thread
            request_id = f"{person_id}_{int(time.time() * 1000)}"
            thread = threading.Thread(
                target=self._make_api_request,
                args=(person_id, request_id, body_image),
                daemon=True
            )
            thread.start()
            
            # Store pending request
            self.pending_requests[person_id] = {
                "thread": thread,
                "request_id": request_id,
                "start_time": time.time(),
                "last_result": {"status": "pending", "description": "Analyzing activity..."}
            }
            
            return self.pending_requests[person_id]["last_result"]
                
        except Exception as e:
            logger.error(f"Error in SmolVLM processing: {e}")
            return {
                "error": str(e),
                "method": "smolvlm_api"
            }
    
    def _make_api_request(self, person_id: int, request_id: str, body_image: np.ndarray) -> None:
        """Make the actual API request in a separate thread."""
        try:
            # Encode image to bytes
            _, img_buffer = cv2.imencode('.jpg', body_image)
            img_bytes = img_buffer.tobytes()
            
            # Prepare files for multipart upload
            files = {'image': ('body_image.jpg', img_bytes, 'image/jpeg')}
            
            # Prepare data payload
            data = {'max_new_tokens': 200}
            
            # Make API request with file upload
            response = requests.post(
                self.api_url,
                headers=self.headers,
                files=files,
                data=data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                description = result.get("description", "No description available")
                
                # Extract the actual description from the API response
                # The API returns "User:\n\n\n\n\nDescribe what the person is doing.\nAssistant: {description}"
                if "Assistant:" in description:
                    actual_description = description.split("Assistant:")[-1].strip()
                else:
                    actual_description = description
                
                # Update pending request with result
                if person_id in self.pending_requests:
                    self.pending_requests[person_id]["last_result"] = {
                        "status": "success",
                        "description": actual_description,
                        "inference_time": result.get("inference_time", 0),
                        "method": "smolvlm_api"
                    }
            else:
                error_msg = f"API request failed: {response.status_code}"
                logger.error(error_msg)
                logger.error(f"Response: {response.text}")
                
                # Update pending request with error
                if person_id in self.pending_requests:
                    self.pending_requests[person_id]["last_result"] = {
                        "error": error_msg,
                        "response": response.text,
                        "method": "smolvlm_api"
                    }
                
        except Exception as e:
            logger.error(f"Error in SmolVLM API request: {e}")
            
            # Update pending request with error
            if person_id in self.pending_requests:
                self.pending_requests[person_id]["last_result"] = {
                    "error": str(e),
                    "method": "smolvlm_api"
                }
        finally:
            # Clean up pending request after a while
            if person_id in self.pending_requests:
                # Keep the result for 30 seconds before removing
                timer = threading.Timer(30.0, self._cleanup_request, args=[person_id])
                timer.daemon = True
                timer.start()
    
    def _cleanup_request(self, person_id: int) -> None:
        """Clean up pending request after some time."""
        if person_id in self.pending_requests:
            del self.pending_requests[person_id]


def create_smolvlm_plugin(api_url: str = "http://localhost:9000/describe", 
                         api_key: str = None,
                         update_interval_ms: int = 5000):
    """Create a SmolVLM activity detection plugin."""
    return SmolVLMPlugin(
        api_url=api_url,
        api_key=api_key,
        update_interval_ms=update_interval_ms
    )