# uv_app/plugins/activity_plugin.py

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


class ActivityPlugin(BasePlugin):
    """Plugin for tracking person activity over time."""
    
    def __init__(self, update_interval_ms: int = 5000):
        super().__init__("activity", update_interval_ms)
        self.person_activities: Dict[int, Dict[str, Any]] = {}
        logger.debug("Initialized ActivityPlugin")
    
    def process_person(self, person, frame) -> Dict[str, Any]:
        """Process person and track their activity."""
        try:
            person_id = person.track_id
            
            # Initialize person activity tracking if needed
            if person_id not in self.person_activities:
                self.person_activities[person_id] = {
                    "first_seen": time.time(),
                    "last_seen": time.time(),
                    "frame_count": 0,
                    "visible_frames": 0,
                    "positions": []
                }
            
            # Update activity tracking
            activity_data = self.person_activities[person_id]
            activity_data["last_seen"] = time.time()
            activity_data["frame_count"] += 1
            
            if person.is_visible:
                activity_data["visible_frames"] += 1
                
                # Store position data
                coords = person.get_current_coordinates()
                if coords["face_bbox"] or coords["body_bbox"]:
                    position_entry = {
                        "timestamp": time.time(),
                        "face_bbox": coords["face_bbox"],
                        "body_bbox": coords["body_bbox"]
                    }
                    activity_data["positions"].append(position_entry)
                    
                    # Keep only recent positions (last 100)
                    if len(activity_data["positions"]) > 100:
                        activity_data["positions"] = activity_data["positions"][-100:]
            
            # Calculate activity metrics
            duration = activity_data["last_seen"] - activity_data["first_seen"]
            visibility_ratio = (activity_data["visible_frames"] / 
                              activity_data["frame_count"]) if activity_data["frame_count"] > 0 else 0
            
            # Movement analysis (simplified)
            movement_score = self._calculate_movement_score(activity_data["positions"])
            
            result = {
                "duration_seconds": round(duration, 2),
                "frame_count": activity_data["frame_count"],
                "visible_frames": activity_data["visible_frames"],
                "visibility_ratio": round(visibility_ratio, 3),
                "movement_score": round(movement_score, 3),
                "position_count": len(activity_data["positions"]),
                "method": "activity_tracking"
            }
            
            logger.debug(f"Tracked activity for person {person_id}")
            return result
            
        except Exception as e:
            error_msg = f"Error tracking activity for person {person.track_id}: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def _calculate_movement_score(self, positions: list) -> float:
        """Calculate a simple movement score based on position changes."""
        if len(positions) < 2:
            return 0.0
        
        # Simple movement calculation based on bounding box center changes
        total_movement = 0.0
        prev_center = None
        
        for pos in positions:
            # Calculate center of bounding box
            bbox = pos.get("face_bbox") or pos.get("body_bbox")
            if bbox:
                if len(bbox) >= 4:
                    center_x = (bbox[1] + bbox[3]) / 2  # (right + left) / 2
                    center_y = (bbox[0] + bbox[2]) / 2  # (top + bottom) / 2
                    current_center = (center_x, center_y)
                    
                    if prev_center:
                        # Calculate distance moved
                        distance = ((current_center[0] - prev_center[0])**2 + 
                                  (current_center[1] - prev_center[1])**2)**0.5
                        total_movement += distance
                    
                    prev_center = current_center
        
        # Normalize by number of position changes
        return total_movement / (len(positions) - 1) if len(positions) > 1 else 0.0