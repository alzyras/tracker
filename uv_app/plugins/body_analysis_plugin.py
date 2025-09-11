# uv_app/plugins/body_analysis_plugin.py

import cv2
import numpy as np
from typing import Dict, Any
from .base import BodyPlugin

# Fix the import issue by using absolute import
try:
    from ..core.logging import get_logger
except (ImportError, ValueError):
    # Fallback to absolute import
    from uv_app.core.logging import get_logger

logger = get_logger()


class BodyAnalysisPlugin(BodyPlugin):
    """Plugin for analyzing body images and extracting descriptive information."""
    
    def __init__(self, update_interval_ms: int = 2000):
        super().__init__("body_analysis", update_interval_ms)
        logger.debug("Initialized BodyAnalysisPlugin")
    
    def process_body(self, body_image: np.ndarray, person) -> Dict[str, Any]:
        """Process body image and extract descriptive information."""
        try:
            # Basic image statistics
            height, width = body_image.shape[:2]
            gray = cv2.cvtColor(body_image, cv2.COLOR_BGR2GRAY)
            brightness = float(np.mean(gray))
            contrast = float(np.std(gray))
            
            # Color analysis
            b_mean, g_mean, r_mean = np.mean(body_image, axis=(0, 1))
            
            # Estimate body proportions (mock implementation)
            proportions = self._estimate_body_proportions(body_image)
            
            # Dominant colors
            dominant_colors = self._extract_dominant_colors(body_image)
            
            # Body posture estimation (mock)
            posture = self._estimate_posture(body_image)
            
            result = {
                "dimensions": {
                    "height": height,
                    "width": width
                },
                "color_stats": {
                    "brightness": brightness,
                    "contrast": contrast,
                    "avg_blue": float(b_mean),
                    "avg_green": float(g_mean),
                    "avg_red": float(r_mean)
                },
                "body_proportions": proportions,
                "dominant_colors": dominant_colors,
                "estimated_posture": posture,
                "method": "body_analysis"
            }
            
            logger.debug(f"Processed body image for person {person.track_id}")
            return result
            
        except Exception as e:
            error_msg = f"Error processing body image: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def _estimate_body_proportions(self, body_image: np.ndarray) -> Dict[str, float]:
        """Estimate body proportions from body image."""
        # Mock implementation - in reality you'd use pose landmarks
        height, width = body_image.shape[:2]
        
        # Simple heuristics based on image dimensions
        if height > width * 1.5:
            body_type = "tall"
        elif width > height * 0.8:
            body_type = "wide"
        else:
            body_type = "average"
        
        return {
            "aspect_ratio": float(height / width) if width > 0 else 0,
            "body_type": body_type
        }
    
    def _extract_dominant_colors(self, body_image: np.ndarray) -> list:
        """Extract dominant colors from body image."""
        # Reshape image to be a list of pixels
        pixels = body_image.reshape((-1, 3))
        
        # Convert to float
        pixels = np.float32(pixels)
        
        # Define criteria and apply k-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
        k = 3
        _, labels, palette = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Get dominant colors
        _, counts = np.unique(labels, return_counts=True)
        dominant_indices = np.argsort(counts)[::-1][:3]
        
        dominant_colors = []
        for i in dominant_indices:
            color = palette[i]
            dominant_colors.append({
                "b": int(color[0]),
                "g": int(color[1]),
                "r": int(color[2]),
                "percentage": float(counts[i] / len(labels))
            })
        
        return dominant_colors
    
    def _estimate_posture(self, body_image: np.ndarray) -> str:
        """Estimate body posture (mock implementation)."""
        # Mock implementation - in reality you'd use pose landmarks
        gray = cv2.cvtColor(body_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Simple heuristic: more vertical lines might indicate standing
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=20, maxLineGap=5)
        
        if lines is not None:
            vertical_lines = 0
            horizontal_lines = 0
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                
                if abs(angle) < 30 or abs(angle) > 150:  # Horizontal
                    horizontal_lines += 1
                elif abs(angle) > 60 and abs(angle) < 120:  # Vertical
                    vertical_lines += 1
            
            if vertical_lines > horizontal_lines:
                return "standing"
            elif horizontal_lines > vertical_lines:
                return "lying"
            else:
                return "sitting"
        else:
            return "unknown"