# uv_app/config.py

import os

# ---------------- CONFIG -----------------
SAVE_DIR = "tracked_people"  # Directory to save and load tracked people data
MAX_MISSED_FRAMES = 50       # Frames a person can be "lost" before forgotten
MAX_FACE_IMAGES = 50         # Max face images to store per person
MIN_FRAMES_TO_CONFIRM = 5    # Frames required to confirm a new person
MATCH_THRESHOLD = 0.6        # Face recognition match threshold (lower = stricter)
CANDIDATE_THRESHOLD = 0.7    # Threshold for considering as candidate
RESIZE_MAX = 640             # Max frame size for faster processing
# ----------------------------------------

# Ensure save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)