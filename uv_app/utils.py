"""Utility functions for frame processing and drawing."""

from typing import Optional, Tuple

import cv2
import numpy as np

from .config import (
    BOX_THICKNESS,
    FACE_BOX_COLOR,
    BODY_BOX_COLOR,
    FONT_FACE,
    FONT_SCALE,
    FONT_THICKNESS,
    FRAME_SCALE,
    RESIZE_MAX,
    SHOW_FACE_BOXES,
    SHOW_BODY_BOXES,
    SHOW_PERSON_ID,
    SHOW_PERSON_NAME,
    TEXT_BACKGROUND_COLOR,
    TEXT_COLOR,
    TEXT_PADDING,
)
from .logging_config import get_logger

logger = get_logger(__name__)


def resize_frame(frame: np.ndarray) -> np.ndarray:
    """Resize frame if larger than max allowed size.
    
    Args:
        frame: Input frame
        
    Returns:
        Resized frame
    """
    h, w = frame.shape[:2]
    if max(h, w) > RESIZE_MAX:
        scale = RESIZE_MAX / max(h, w)
        return cv2.resize(frame, (int(w * scale), int(h * scale)))
    return frame


def scale_frame(frame: np.ndarray, scale: Optional[float] = None) -> np.ndarray:
    """Scale frame by the configured scale factor.
    
    Args:
        frame: Input frame
        scale: Scale factor (uses FRAME_SCALE if None)
        
    Returns:
        Scaled frame
    """
    scale_factor = scale if scale is not None else FRAME_SCALE
    if scale_factor != 1.0:
        h, w = frame.shape[:2]
        new_w, new_h = int(w * scale_factor), int(h * scale_factor)
        return cv2.resize(frame, (new_w, new_h))
    return frame


def draw_text_with_background(
    frame: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font_face: int = FONT_FACE,
    font_scale: float = FONT_SCALE,
    text_color: Tuple[int, int, int] = TEXT_COLOR,
    bg_color: Tuple[int, int, int] = TEXT_BACKGROUND_COLOR,
    thickness: int = FONT_THICKNESS,
    padding: int = TEXT_PADDING,
) -> None:
    """Draw text with background rectangle.
    
    Args:
        frame: Frame to draw on
        text: Text to draw
        position: (x, y) position for text
        font_face: OpenCV font face
        font_scale: Font scale
        text_color: Text color (BGR)
        bg_color: Background color (BGR)
        thickness: Text thickness
        padding: Background padding
    """
    x, y = position
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font_face, font_scale, thickness
    )
    
    # Draw background rectangle
    cv2.rectangle(
        frame,
        (x - padding, y - text_height - padding),
        (x + text_width + padding, y + baseline + padding),
        bg_color,
        -1,
    )
    
    # Draw text
    cv2.putText(
        frame, text, (x, y), font_face, font_scale, text_color, thickness
    )


def draw_face_box(
    frame: np.ndarray,
    box: Tuple[int, int, int, int],
    label: str,
    color: Optional[Tuple[int, int, int]] = None,
) -> None:
    """Draw face bounding box and label on frame.
    
    Args:
        frame: Frame to draw on
        box: Bounding box (top, right, bottom, left)
        label: Label text
        color: Box color (uses FACE_BOX_COLOR if None)
    """
    if not SHOW_FACE_BOXES:
        return
        
    top, right, bottom, left = box
    box_color = color if color is not None else FACE_BOX_COLOR
    
    # Draw bounding box
    cv2.rectangle(frame, (left, top), (right, bottom), box_color, BOX_THICKNESS)
    
    # Draw label with background
    if label:
        draw_text_with_background(frame, label, (left, top - 10))


def draw_body_box(
    frame: np.ndarray,
    box: Tuple[int, int, int, int],
    label: str,
    color: Optional[Tuple[int, int, int]] = None,
) -> None:
    """Draw body bounding box and label on frame.
    
    Args:
        frame: Frame to draw on
        box: Bounding box (x1, y1, x2, y2)
        label: Label text
        color: Box color (uses BODY_BOX_COLOR if None)
    """
    if not SHOW_BODY_BOXES:
        return
        
    x1, y1, x2, y2 = box
    box_color = color if color is not None else BODY_BOX_COLOR
    
    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, BOX_THICKNESS)
    
    # Draw label with background
    if label:
        draw_text_with_background(frame, label, (x1, y1 - 10))


def create_person_label(
    person_id: int,
    person_name: Optional[str] = None,
    additional_info: Optional[str] = None,
) -> str:
    """Create a label for a tracked person.
    
    Args:
        person_id: Person's track ID
        person_name: Person's name (if available)
        additional_info: Additional information to display
        
    Returns:
        Formatted label string
    """
    parts = []
    
    if SHOW_PERSON_ID:
        parts.append(f"ID {person_id}")
    
    if SHOW_PERSON_NAME and person_name:
        parts.append(f"({person_name})")
    
    if additional_info:
        parts.append(additional_info)
    
    return " ".join(parts) if parts else str(person_id)
