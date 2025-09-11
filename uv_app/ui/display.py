# uv_app/ui/display.py

import cv2
import numpy as np
from typing import Tuple, Optional

# Lazy import of config loader to avoid circulars in some run modes
try:
    from ..config_loader import config_loader  # type: ignore
except Exception:
    # Fallback to absolute import if package context differs
    try:
        from uv_app.config_loader import config_loader  # type: ignore
    except Exception:
        config_loader = None  # type: ignore


def _hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    """Convert a hex color like '#RRGGBB' to OpenCV BGR tuple."""
    if not isinstance(hex_color, str):
        return (0, 255, 0)
    h = hex_color.lstrip('#')
    if len(h) != 6:
        return (0, 255, 0)
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return (b, g, r)


def _get_line_type(style: str) -> int:
    """Map style to OpenCV line type; custom dash/dot handled separately."""
    return cv2.LINE_AA


def _get_font(family: str) -> int:
    """Map font family to closest OpenCV font."""
    fam = (family or '').lower()
    if 'courier' in fam or 'mono' in fam:
        return cv2.FONT_HERSHEY_PLAIN
    if 'times' in fam or 'serif' in fam:
        return cv2.FONT_HERSHEY_COMPLEX
    # Arial/Helvetica/Verdana → SIMPLEX
    return cv2.FONT_HERSHEY_SIMPLEX


def _get_ui() -> dict:
    if config_loader is None:
        return {}
    try:
        return config_loader.get_ui_settings() or {}
    except Exception:
        return {}


def _draw_styled_rect(frame: np.ndarray,
                      pt1: Tuple[int, int],
                      pt2: Tuple[int, int],
                      color: Tuple[int, int, int],
                      thickness: int,
                      style: str) -> None:
    """Draw rectangle with optional dashed/dotted style."""
    style_norm = (style or 'Solid').lower()
    if style_norm == 'solid':
        cv2.rectangle(frame, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)
        return

    # For dashed/dotted, draw 4 sides as line segments
    x1, y1 = pt1
    x2, y2 = pt2
    segments = 6 if style_norm == 'dashed' else 20
    # Horizontal top and bottom
    for i in range(segments):
        t0 = i / segments
        t1 = (i + 0.5) / segments if style_norm == 'dashed' else (i + 0.2) / segments
        if t1 > 1:
            break
        # top
        p0 = (int(x1 + (x2 - x1) * t0), y1)
        p1 = (int(x1 + (x2 - x1) * t1), y1)
        cv2.line(frame, p0, p1, color, thickness, lineType=cv2.LINE_AA)
        # bottom
        p0b = (int(x1 + (x2 - x1) * t0), y2)
        p1b = (int(x1 + (x2 - x1) * t1), y2)
        cv2.line(frame, p0b, p1b, color, thickness, lineType=cv2.LINE_AA)
    # Vertical left and right
    for i in range(segments):
        t0 = i / segments
        t1 = (i + 0.5) / segments if style_norm == 'dashed' else (i + 0.2) / segments
        if t1 > 1:
            break
        # left
        p0 = (x1, int(y1 + (y2 - y1) * t0))
        p1 = (x1, int(y1 + (y2 - y1) * t1))
        cv2.line(frame, p0, p1, color, thickness, lineType=cv2.LINE_AA)
        # right
        p0r = (x2, int(y1 + (y2 - y1) * t0))
        p1r = (x2, int(y1 + (y2 - y1) * t1))
        cv2.line(frame, p0r, p1r, color, thickness, lineType=cv2.LINE_AA)


class FaceDisplayManager:
    """Handles face bounding box display and labeling."""
    
    @staticmethod
    def draw_face_box(frame: np.ndarray, box: Tuple[int, int, int, int], 
                     label: str, color: Tuple[int, int, int] = None) -> None:
        """
        Draw bounding box and label on frame.
        
        Args:
            frame: Input frame
            box: Face location tuple (top, right, bottom, left)
            label: Text label to display
            color: BGR color tuple
        """
        ui = _get_ui()
        top, right, bottom, left = box
        box_color = _hex_to_bgr(ui.get('box_color', '#00FF00')) if color is None else color
        border_width = int(ui.get('border_width', 2))
        box_style = ui.get('box_style', 'Solid')
        font_family = ui.get('font_family', 'Arial')
        font_color = _hex_to_bgr(ui.get('font_color', '#FFFFFF'))
        font_size = float(ui.get('font_size', 12))
        # Map font size to OpenCV scale (heuristic: 12pt ~ 0.5 scale)
        font_scale = max(0.3, (font_size / 12.0) * 0.5)
        font = _get_font(font_family)

        _draw_styled_rect(frame, (left, top), (right, bottom), box_color, border_width, box_style)
        cv2.putText(frame, label, (left, max(0, top - 10)), font, font_scale, font_color, max(1, border_width // 2), lineType=cv2.LINE_AA)
    
    @staticmethod
    def draw_face_box_with_emotion(frame: np.ndarray, box: Tuple[int, int, int, int], 
                                  label: str, emotion: Optional[str] = None,
                                  probability: Optional[float] = None,
                                  color: Tuple[int, int, int] = None) -> None:
        """
        Draw bounding box, label, and emotion on frame.
        
        Args:
            frame: Input frame
            box: Face location tuple (top, right, bottom, left)
            label: Text label to display
            emotion: Emotion text to display below the box
            color: BGR color tuple
        """
        ui = _get_ui()
        top, right, bottom, left = box
        box_color = _hex_to_bgr(ui.get('box_color', '#00FF00')) if color is None else color
        border_width = int(ui.get('border_width', 2))
        box_style = ui.get('box_style', 'Solid')
        font_family = ui.get('font_family', 'Arial')
        font_color = _hex_to_bgr(ui.get('font_color', '#FFFFFF'))
        font_size = float(ui.get('font_size', 12))
        font_scale = max(0.3, (font_size / 12.0) * 0.5)
        font = _get_font(font_family)

        _draw_styled_rect(frame, (left, top), (right, bottom), box_color, border_width, box_style)
        cv2.putText(frame, label, (left, max(0, top - 10)), font, font_scale, font_color, max(1, border_width // 2), lineType=cv2.LINE_AA)
        
        # Draw emotion below the box if available
        if emotion:
            # Normalize and abbreviate to 3 letters
            em = emotion.lower() if isinstance(emotion, str) else ""
            # Canonical mapping for abbreviations
            abbrev_map = {
                'happiness': 'hap', 'happy': 'hap',
                'anger': 'ang', 'angry': 'ang',
                'sadness': 'sad', 'sad': 'sad',
                'surprise': 'sur', 'surprised': 'sur',
                'neutral': 'neu',
                'fear': 'fea',
                'disgust': 'dis',
                'contempt': 'con'
            }
            abbrev = abbrev_map.get(em, (em[:3] if em else ''))
            # Emoji mapping (may not render on all systems)
            emoji_map = {
                'hap': '😀',
                'neu': '😐',
                'sad': '😢',
                'ang': '😠',
                'sur': '😮',
                'fea': '😱',
                'dis': '🤢',
                'con': '🙄'
            }
            emoji = emoji_map.get(abbrev, '')
            if probability is not None:
                emotion_text = f"{emoji} {abbrev} ({probability:.2f})" if emoji else f"{abbrev} ({probability:.2f})"
            else:
                emotion_text = f"{emoji} {abbrev}" if emoji else f"{abbrev}"
            cv2.putText(frame, emotion_text, (left, bottom + 20), 
                       font, max(0.4, font_scale), font_color, max(1, border_width // 2), lineType=cv2.LINE_AA)
    
    @staticmethod
    def draw_body_box(frame: np.ndarray, box: Tuple[int, int, int, int], 
                     label: Optional[str] = None, color: Tuple[int, int, int] = None) -> None:
        """
        Draw body bounding box on frame.
        
        Args:
            frame: Input frame
            box: Body location tuple (x1, y1, x2, y2)
            label: Optional text label
            color: BGR color tuple
        """
        ui = _get_ui()
        x1, y1, x2, y2 = box
        box_color = _hex_to_bgr(ui.get('box_color', '#00FF00')) if color is None else color
        border_width = int(ui.get('border_width', 2))
        box_style = ui.get('box_style', 'Solid')
        font_family = ui.get('font_family', 'Arial')
        font_color = _hex_to_bgr(ui.get('font_color', '#FFFFFF'))
        font_size = float(ui.get('font_size', 12))
        font_scale = max(0.3, (font_size / 12.0) * 0.5)
        font = _get_font(font_family)

        _draw_styled_rect(frame, (x1, y1), (x2, y2), box_color, border_width, box_style)
        if label:
            cv2.putText(frame, label, (x1, max(0, y1 - 10)), font, font_scale, font_color, max(1, border_width // 2), lineType=cv2.LINE_AA)
    
    @staticmethod
    def draw_info_panel(frame: np.ndarray, info: dict) -> None:
        """
        Draw information panel on frame.
        
        Args:
            frame: Input frame
            info: Dictionary of information to display
        """
        y_offset = 30
        for key, value in info.items():
            text = f"{key}: {value}"
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25


class VideoDisplayManager:
    """Handles video display and window management."""
    
    def __init__(self, window_name: str = "Tracker"):
        self.window_name = window_name
        self.is_window_created = False
    
    def create_window(self) -> None:
        """Create the display window."""
        if not self.is_window_created:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            self.is_window_created = True
    
    def show_frame(self, frame: np.ndarray) -> None:
        """
        Display a frame in the window.
        
        Args:
            frame: Frame to display
        """
        self.create_window()
        cv2.imshow(self.window_name, frame)
    
    def wait_for_key(self, timeout: int = 1) -> int:
        """
        Wait for a key press.
        
        Args:
            timeout: Timeout in milliseconds
            
        Returns:
            Key code or -1 if timeout
        """
        return cv2.waitKey(timeout) & 0xFF
    
    def should_exit(self, key: int) -> bool:
        """
        Check if the application should exit.
        
        Args:
            key: Key code from waitKey
            
        Returns:
            True if should exit
        """
        return key in (27, ord('q'))  # ESC or 'q'
    
    def destroy_windows(self) -> None:
        """Destroy all OpenCV windows."""
        cv2.destroyAllWindows()
    
    def resize_frame(self, frame: np.ndarray, max_size: int = 640) -> np.ndarray:
        """
        Resize frame if larger than max size.
        
        Args:
            frame: Input frame
            max_size: Maximum dimension size
            
        Returns:
            Resized frame
        """
        h, w = frame.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            return cv2.resize(frame, (int(w * scale), int(h * scale)))
        return frame
