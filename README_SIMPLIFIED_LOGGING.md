# Simplified Logging System

This document explains the changes made to reduce verbose logging and focus on important events.

## Changes Made

### 1. Reduced Verbose Logging
- **Disabled Spam**: Turned off verbose face match logging, detection logging, and plugin results logging
- **Focused Output**: Now only shows important tracking events
- **Configuration**: Controlled through `LOGGING_CONFIG` in `config.py`
- **VERBOSE_TRACKING**: Set to `False` to completely disable detailed tracking messages

### 2. Added Person Entry/Exit Tracking
- Created `PersonEventLogger` plugin to track when people enter/exit the stream
- Logs entry events: `🚪 Person ID 1 entered the stream at 15:30:45`
- Logs exit events: `🚪 Person ID 1 left the stream at 15:35:22 (Duration: 277.0 seconds)`

### 3. Enhanced Emotion Logging
- Consolidated emotion information into single log lines
- Format: `😊 Emotions: Person ID 1 is happy (0.85), Agne is neutral (0.74)`
- Logs every 5 seconds instead of after every frame

## Configuration

The simplified logging is controlled through `config.py`:

```python
# Logging configuration
LOGGING_CONFIG = {
    'log_level': 'INFO',
    'log_matches': False,  # Disable verbose face match logging
    'log_detections': False,  # Disable verbose detection logging
    'log_tracking_events': True,  # Keep important tracking events
    'log_plugin_results': False,  # Disable verbose plugin results
    'enable_file_logging': True,
}

# Disable verbose tracking messages
VERBOSE_TRACKING = False

# Plugin configuration
PLUGIN_CONFIG = {
    'emotion_logger_enabled': True,
    'emotion_logger_interval': 5000,  # milliseconds (5 seconds)
    'person_event_logger_enabled': True,
    'person_event_logger_interval': 1000,  # milliseconds (1 second)
}
```

## Example Output

With the new system, you'll see logs like:

```
2025-09-10 15:30:45,123 [INFO] - 🚪 Person ID 1 entered the stream at 15:30:45
2025-09-10 15:30:47,456 [INFO] - 😊 Emotions: Person ID 1 is happy (0.85)
2025-09-10 15:30:52,789 [INFO] - 😊 Emotions: Person ID 1 is happy (0.82)
2025-09-10 15:35:22,345 [INFO] - 🚪 Person ID 1 left the stream at 15:35:22 (Duration: 277.0 seconds)
```

This provides all the important information without the verbose spam of face match details.

## Controlling Verbosity

To re-enable detailed tracking messages, set:
```python
VERBOSE_TRACKING = True
```

To re-enable specific types of logging, modify the `LOGGING_CONFIG`:
```python
LOGGING_CONFIG = {
    'log_matches': True,      # Enable face match logging
    'log_detections': True,   # Enable detection logging
    'log_plugin_results': True,  # Enable plugin results logging
}
```