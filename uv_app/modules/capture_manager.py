"""Capture manager for saving snapshots at configured intervals."""

import time
from typing import Dict, Optional

import cv2
import numpy as np

from ..logging_config import get_logger
from ..person import TrackedPerson

logger = get_logger(__name__)


class CaptureManager:
    """Saves head, body, and pose snapshots based on configured intervals."""

    def __init__(self, intervals: Dict[str, Optional[float]]) -> None:
        """Initialize the capture manager.
        
        Args:
            intervals: Dictionary mapping capture types to intervals in seconds
        """
        self.intervals = intervals
        self._last_capture_time: Dict[str, float] = {
            k: 0.0 for k in ("head", "body", "pose")
        }
        logger.info("Initialized CaptureManager with intervals: %s", intervals)

    def _should_capture(self, kind: str) -> bool:
        """Check if it's time to capture a snapshot of the given type.
        
        Args:
            kind: Type of capture (head, body, pose)
            
        Returns:
            True if capture should happen
        """
        interval = self.intervals.get(kind)
        if not interval or interval <= 0:
            return False
        now = time.time()
        if now - self._last_capture_time.get(kind, 0.0) >= interval:
            self._last_capture_time[kind] = now
            return True
        return False

    def maybe_capture_head(self, person: TrackedPerson, frame_bgr: np.ndarray) -> None:
        """Capture head snapshot if interval has elapsed.
        
        Args:
            person: The tracked person
            frame_bgr: Current frame in BGR format
        """
        if not self._should_capture("head"):
            return
        if person.face_boxes:
            top, right, bottom, left = person.face_boxes[-1]
            crop = frame_bgr[top:bottom, left:right]
            if crop.size > 0:
                person.save_snapshot(crop, "head")
                logger.debug("Captured head snapshot for person %d", person.track_id)

    def maybe_capture_body(self, person: TrackedPerson, frame_bgr: np.ndarray) -> None:
        """Capture body snapshot if interval has elapsed.
        
        Args:
            person: The tracked person
            frame_bgr: Current frame in BGR format
        """
        if not self._should_capture("body"):
            return
        if person.body_boxes:
            x1, y1, x2, y2 = person.body_boxes[-1]
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size > 0:
                person.save_snapshot(crop, "body")
                logger.debug("Captured body snapshot for person %d", person.track_id)

    def maybe_capture_pose(self, person: TrackedPerson, frame_bgr: np.ndarray) -> None:
        """Capture pose snapshot if interval has elapsed.
        
        Args:
            person: The tracked person
            frame_bgr: Current frame in BGR format
        """
        if not self._should_capture("pose"):
            return
        # For pose, save the full frame
        person.save_snapshot(frame_bgr, "pose")
        logger.debug("Captured pose snapshot for person %d", person.track_id)