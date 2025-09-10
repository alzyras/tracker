# uv_app/config.py

import os

# ---------------- CONFIG -----------------
SAVE_DIR = "tracked_people"  # Directory to save and load tracked people data
MAX_MISSED_FRAMES = 50       # Frames a person can be "lost" before forgotten
MAX_FACE_IMAGES = 30         # Max face images to store per person
MIN_FRAMES_TO_CONFIRM = 5    # Frames required to confirm a new person
MATCH_THRESHOLD = 0.45       # Face recognition match threshold (lower = stricter)
CANDIDATE_THRESHOLD = 0.4    # Threshold for considering as candidate (stricter than match)
RESIZE_MAX = 640             # Max frame size for faster processing
VERBOSE_TRACKING = False     # Show detailed tracking messages (set to False to reduce verbosity)

# Logging configuration
LOGGING_CONFIG = {
    'log_level': 'INFO',
    'log_matches': False,  # Disable verbose face match logging
    'log_detections': False,  # Disable verbose detection logging
    'log_tracking_events': True,  # Keep important tracking events
    'log_plugin_results': False,  # Disable verbose plugin results
    'enable_file_logging': True,
}

# Plugin configuration
PLUGIN_CONFIG = {
    'emotion_plugin_enabled': False,  # Disable local emotion plugin
    'simple_emotion_enabled': False,  # Disable simple emotion plugin
    'api_emotion_plugin_enabled': True,  # Enable API-based emotion plugin
    'api_emotion_plugin_interval': 2000,  # milliseconds
    'api_emotion_api_url': 'http://localhost:8080',  # Emotion API URL
    'activity_plugin_enabled': True,
    'activity_plugin_interval': 5000,  # milliseconds
    'emotion_logger_enabled': True,
    'emotion_logger_interval': 5000,  # milliseconds (5 seconds)
    'person_event_logger_enabled': True,
    'person_event_logger_interval': 1000,  # milliseconds (1 second)
}
# ----------------------------------------

# Ensure save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)