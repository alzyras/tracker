"""Configuration settings for the tracking application."""

import os
from typing import Tuple

# ---------------- CORE TRACKING CONFIG -----------------
SAVE_DIR = "tracked_people"  # Directory to save and load tracked people data
MAX_MISSED_FRAMES = 50       # Frames a person can be "lost" before forgotten
MAX_FACE_IMAGES = 50         # Max face images to store per person
MIN_FRAMES_TO_CONFIRM = 5    # Frames required to confirm a new person
MATCH_THRESHOLD = 0.5        # Strict match threshold
CANDIDATE_THRESHOLD = 0.6    # More forgiving candidate threshold

# ---------------- DISPLAY CONFIG -----------------
# Frame processing
FRAME_SCALE = 1.0            # Scale factor for display (1.0 = original size)
RESIZE_MAX = 640             # Max frame size for faster processing

# Bounding box colors (BGR format)
FACE_BOX_COLOR: Tuple[int, int, int] = (0, 255, 0)    # Green
BODY_BOX_COLOR: Tuple[int, int, int] = (255, 0, 0)    # Blue
POSE_COLOR: Tuple[int, int, int] = (0, 0, 255)        # Red
POSE_CONNECTION_COLOR: Tuple[int, int, int] = (255, 255, 0)  # Cyan

# Bounding box settings
BOX_THICKNESS = 2
BOX_ALPHA = 0.3              # Transparency for filled boxes (0.0-1.0)

# Text settings
FONT_FACE = 0                # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2
TEXT_COLOR: Tuple[int, int, int] = (255, 255, 255)    # White
TEXT_BACKGROUND_COLOR: Tuple[int, int, int] = (0, 0, 0)  # Black
TEXT_PADDING = 5

# Display options
SHOW_FACE_BOXES = True
SHOW_BODY_BOXES = True
SHOW_POSES = True
SHOW_PERSON_ID = True
SHOW_PERSON_NAME = True
SHOW_CONFIDENCE = False

# Pose visualization
POSE_LANDMARK_RADIUS = 2
POSE_CONNECTION_THICKNESS = 2

# ---------------- LOGGING CONFIG -----------------
LOG_LEVEL = "INFO"
LOG_TO_FILE = True
LOG_FILE_PATH = "tracker.log"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
MAX_LOG_FILE_SIZE = 10 * 1024 * 1024  # 10MB
LOG_BACKUP_COUNT = 5

# ---------------- ANALYZER CONFIG -----------------
ENABLE_ANALYZERS = True
ANALYZER_LOG_RESULTS = True

# Capture intervals for snapshots (in seconds, None to disable)
CAPTURE_INTERVALS = {
    "head": 5.0,    # Capture head every 5 seconds
    "body": 10.0,   # Capture body every 10 seconds
    "pose": 15.0,   # Capture pose every 15 seconds
}

# ----------------------------------------

# Ensure save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)