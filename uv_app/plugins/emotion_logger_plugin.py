# uv_app/plugins/emotion_logger_plugin.py

import time
from typing import Dict, Any
from .base import BasePlugin

# Fix the import issue by using absolute import
try:
    from ..core.logging import get_logger
except (ImportError, ValueError):
    # Fallback to absolute import
    from uv_app.core.logging import get_logger

logger = get_logger()


class EmotionLoggerPlugin(BasePlugin):
    """Plugin for logging emotion information every 5 seconds."""
    
    def __init__(self, update_interval_ms: int = 5000):
        super().__init__("emotion_logger", update_interval_ms)
        self.last_log_time = 0
        logger.debug("Initialized EmotionLoggerPlugin")
    
    def process_person(self, person, frame) -> Dict[str, Any]:
        """Process person and log emotion information if available."""
        try:
            # This plugin doesn't process the person directly, but logs stored emotion data
            return {"status": "logging_ready"}
        except Exception as e:
            error_msg = f"Error in emotion logging: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def log_emotions(self, people_list) -> None:
        """Log emotion information for all visible people."""
        try:
            current_time_ms = int(time.time() * 1000)
            
            # Only log every 5 seconds
            if current_time_ms - self.last_log_time >= self.update_interval_ms:
                emotion_messages = []
                for person in people_list:
                    if not person.is_visible:
                        continue
                    
                    # Get emotion from plugin results
                    emotion_info = self._get_person_emotion(person)
                    if emotion_info:
                        person_name = person.name if person.name else f"Person ID {person.track_id}"
                        emotion_messages.append(f"{person_name} is {emotion_info['emotion']} ({emotion_info['confidence']:.2f})")
                
                if emotion_messages:
                    logger.info(f"😊 Emotions: {', '.join(emotion_messages)}")
                
                self.last_log_time = current_time_ms
                
        except Exception as e:
            logger.error(f"Error logging emotions: {e}")
    
    def _get_person_emotion(self, person) -> Dict[str, Any]:
        """Get emotion information for a person."""
        # This would typically be called by the tracking system to get emotion data
        # For now, we'll return None as this plugin is primarily for logging
        return None