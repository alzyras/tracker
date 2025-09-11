# uv_app/plugins/face_image_plugin.py

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


class FaceImagePlugin(FacePlugin):
    """Plugin for processing face images to extract detailed information."""
    
    def __init__(self, update_interval_ms: int = 1000):
        super().__init__("face_image", update_interval_ms)
        logger.debug("Initialized FaceImagePlugin")
    
    def process_face(self, face_image: np.ndarray, person) -> Dict[str, Any]:
        """Process face image and extract detailed information."""
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            
            # Basic image statistics
            height, width = face_image.shape[:2]
            brightness = float(np.mean(gray))
            contrast = float(np.std(gray))
            
            # Color analysis
            b_mean, g_mean, r_mean = np.mean(face_image, axis=(0, 1))
            
            # Face quality metrics
            sharpness = self._calculate_sharpness(gray)
            blur_score = self._calculate_blur_score(gray)
            
            # Estimate age and gender (mock implementation)
            age, gender = self._estimate_age_gender(face_image)
            
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
                "quality_metrics": {
                    "sharpness": sharpness,
                    "blur_score": blur_score
                },
                "estimated_demographics": {
                    "age": age,
                    "gender": gender
                },
                "method": "face_image_analysis"
            }
            
            logger.debug(f"Processed face image for person {person.track_id}")
            return result
            
        except Exception as e:
            error_msg = f"Error processing face image: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def _calculate_sharpness(self, gray_image: np.ndarray) -> float:
        """Calculate image sharpness using Laplacian variance."""
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        return float(laplacian.var())
    
    def _calculate_blur_score(self, gray_image: np.ndarray) -> float:
        """Calculate blur score using FFT."""
        # Convert to float
        image_float = np.float64(gray_image)
        
        # Calculate FFT
        fft = np.fft.fft2(image_float)
        fft_shift = np.fft.fftshift(fft)
        
        # Calculate magnitude spectrum
        magnitude_spectrum = np.abs(fft_shift)
        
        # Calculate blur score (simplified)
        # Higher values indicate more high-frequency content (sharper image)
        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2
        
        # Focus on high-frequency regions (away from center)
        mask = np.ones((h, w), dtype=bool)
        y, x = np.ogrid[:h, :w]
        mask_area = (y - center_h)**2 + (x - center_w)**2 <= (min(h, w) // 4)**2
        mask[mask_area] = False
        
        high_freq_content = magnitude_spectrum[mask]
        blur_score = np.mean(high_freq_content) if high_freq_content.size > 0 else 0.0
        
        return float(blur_score)
    
    def _estimate_age_gender(self, face_image: np.ndarray) -> tuple:
        """Mock age and gender estimation."""
        # Simple mock based on image properties
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        # Very basic mock - in reality, you'd use a trained model
        if brightness > 140:
            age = np.random.randint(20, 35)
            gender = "female"
        else:
            age = np.random.randint(25, 50)
            gender = "male"
        
        return int(age), gender