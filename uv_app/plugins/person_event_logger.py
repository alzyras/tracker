# uv_app/plugins/person_event_logger.py

import time
from typing import Dict, Any
from .base import BasePlugin
from ..core.logging import get_logger

logger = get_logger()


class PersonEventLogger(BasePlugin):
    """Plugin for logging person entry/exit events and tracking duration."""
    
    def __init__(self, update_interval_ms: int = 1000):
        super().__init__("person_event_logger", update_interval_ms)
        self.person_entries = {}  # track_id -> entry_time
        self.person_durations = {}  # track_id -> duration_seconds
        logger.debug("Initialized PersonEventLogger")
    
    def process_person(self, person, frame) -> Dict[str, Any]:
        """Process person and log entry/exit events."""
        try:
            current_time = time.time()
            
            # Check if this is a new person entry
            if person.track_id not in self.person_entries:
                # This is a new entry
                self.person_entries[person.track_id] = current_time
                person_name = person.name if person.name else f"Person ID {person.track_id}"
                logger.info(f"🚪 {person_name} entered the stream at {time.strftime('%H:%M:%S')}")
            
            # Update duration tracking
            if person.track_id in self.person_entries:
                duration = current_time - self.person_entries[person.track_id]
                self.person_durations[person.track_id] = duration
            
            return {"status": "tracking"}
        except Exception as e:
            error_msg = f"Error in person event logging: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def log_person_exit(self, person) -> None:
        """Log when a person exits the stream."""
        try:
            if person.track_id in self.person_entries:
                entry_time = self.person_entries[person.track_id]
                exit_time = time.time()
                duration = exit_time - entry_time
                
                person_name = person.name if person.name else f"Person ID {person.track_id}"
                logger.info(f"🚪 {person_name} left the stream at {time.strftime('%H:%M:%S')} (Duration: {duration:.1f} seconds)")
                
                # Clean up tracking
                del self.person_entries[person.track_id]
                if person.track_id in self.person_durations:
                    del self.person_durations[person.track_id]
        except Exception as e:
            logger.error(f"Error logging person exit: {e}")
    
    def get_person_duration(self, track_id: int) -> float:
        """Get current duration for a person."""
        return self.person_durations.get(track_id, 0.0)
    
    def get_all_durations(self) -> Dict[int, float]:
        """Get all current person durations."""
        return self.person_durations.copy()