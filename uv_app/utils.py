# uv_app/utils.py

import cv2
import numpy as np
from config import RESIZE_MAX


def resize_frame(frame: np.ndarray) -> np.ndarray:
    """Resize frame if larger than max allowed size."""
    h, w = frame.shape[:2]
    if max(h, w) > RESIZE_MAX:
        scale = RESIZE_MAX / max(h, w)
        return cv2.resize(frame, (int(w * scale), int(h * scale)))
    return frame


def draw_face_box(frame: np.ndarray, box: tuple[int, int, int, int], label: str, color=(0, 255, 0)) -> None:
    """Draw bounding box and label on frame."""
    top, right, bottom, left = box
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
    cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
