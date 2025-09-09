# UV App - Person Tracking Application

A comprehensive person tracking application with face recognition, pose analysis, and configurable analyzers.

## Features

- **Face Recognition**: Track individuals using face encodings
- **Pose Analysis**: Detect body poses and actions using MediaPipe
- **Configurable Display**: Customizable bounding boxes, colors, fonts, and display options
- **Analyzer Framework**: Extensible system for adding custom analyzers
- **Comprehensive Logging**: File and console logging with analyzer result tracking
- **Snapshot Capture**: Automatic capture of head, body, and pose snapshots
- **Persistent Storage**: Save and load tracked person data

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python -m uv_app
```

## Configuration

All configuration options are available in `uv_app/config.py`:

### Display Configuration
- `FRAME_SCALE`: Scale factor for display (1.0 = original size)
- `FACE_BOX_COLOR`, `BODY_BOX_COLOR`, `POSE_COLOR`: Bounding box colors (BGR format)
- `BOX_THICKNESS`: Thickness of bounding boxes
- `FONT_SCALE`, `FONT_THICKNESS`: Text display settings
- `SHOW_FACE_BOXES`, `SHOW_BODY_BOXES`, `SHOW_POSES`: Toggle display elements

### Logging Configuration
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `LOG_TO_FILE`: Enable/disable file logging
- `LOG_FILE_PATH`: Path to log file
- `ANALYZER_LOG_RESULTS`: Log analyzer results

### Analyzer Configuration
- `ENABLE_ANALYZERS`: Enable/disable analyzer system
- `CAPTURE_INTERVALS`: Snapshot capture intervals in seconds

## Adding Custom Analyzers

The application includes an extensible analyzer framework. Here's how to add a custom analyzer:

### 1. Create Your Analyzer Class

Create a new file in `uv_app/modules/analyzers/` (e.g., `my_analyzer.py`):

```python
from typing import Any, Dict
import numpy as np
from .base import BaseAnalyzer
from ...person import TrackedPerson

class MyAnalyzer(BaseAnalyzer):
    def __init__(self) -> None:
        super().__init__("my_analyzer")
        # Initialize your analyzer here
        
    def analyze(
        self, 
        person: TrackedPerson, 
        frame: np.ndarray,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Analyze a person in a frame."""
        # Your analysis logic here
        result = {
            "detected": True,
            "confidence": 0.95,
            "custom_data": "example"
        }
        return result
    
    def get_config(self) -> Dict[str, Any]:
        """Get analyzer configuration."""
        return {
            "name": self.name,
            "type": "custom",
            "uses_face": True,  # Set based on what your analyzer needs
            "uses_body": False,
            "uses_pose": False,
        }
    
    def should_analyze(self, person: TrackedPerson) -> bool:
        """Check if this person should be analyzed."""
        # Add conditions for when to run analysis
        return bool(person.face_boxes)  # Example: only if face is detected
```

### 2. Register Your Analyzer

In `uv_app/tracker.py`, add your analyzer to the initialization:

```python
from .modules.analyzers.my_analyzer import MyAnalyzer

# In the run_tracker function, add:
if ENABLE_ANALYZERS:
    analyzer_manager = AnalyzerManager()
    analyzer_manager.register_analyzer(EmotionAnalyzer())
    analyzer_manager.register_analyzer(PoseAnalyzer())
    analyzer_manager.register_analyzer(MyAnalyzer())  # Add your analyzer
```

### 3. Analyzer Results

Your analyzer results will be:
- Automatically logged if `ANALYZER_LOG_RESULTS` is enabled
- Available in the analyzer manager's results
- Logged with person ID and analyzer name

## Built-in Analyzers

### Emotion Analyzer
- Uses FER (Facial Expression Recognition)
- Detects emotions from face regions
- Returns dominant emotion and confidence scores

### Pose Analyzer
- Uses MediaPipe pose landmarks
- Detects common actions: hands_raised, waving, using_phone
- Returns pose quality and detected actions

## Usage Examples

### Basic Usage
```python
from uv_app import run_tracker

# Run with webcam
run_tracker(
    video_source=0,
    enable_face=True,
    enable_body=True,
    enable_pose=True
)
```

### Custom Video Source
```python
# Run with video file
run_tracker(video_source="path/to/video.mp4")

# Run with RTSP stream
run_tracker(video_source="rtsp://camera_ip/stream")
```

### Custom Configuration
```python
from uv_app.config import *
from uv_app import run_tracker

# Modify configuration
FACE_BOX_COLOR = (255, 0, 0)  # Red boxes
SHOW_POSES = False  # Hide pose visualization
LOG_LEVEL = "DEBUG"  # Verbose logging

run_tracker(video_source=0)
```

## Logging

The application provides comprehensive logging:

### Console Output
- Real-time status updates
- Person detection/loss events
- Analyzer results (if enabled)

### File Logging
- Rotating log files (configurable size and count)
- Structured analyzer result logging
- Error tracking and debugging information

### Log Format
```
2024-01-01 12:00:00 [INFO] tracker.main: Starting tracking application
2024-01-01 12:00:01 [INFO] tracker.analyzer.emotion: Analyzer result - Person: 1, Analyzer: emotion, Result: {'dominant_emotion': 'happy', 'confidence': 0.85}
```

## Data Storage

- **Tracked People**: Stored in `tracked_people/person_<ID>/`
- **Face Encodings**: Saved as NumPy arrays
- **Snapshots**: Automatically captured based on configured intervals
- **Metadata**: JSON files with person information

## Keyboard Controls

- **ESC** or **Q**: Exit the application
- The application automatically saves all data on exit

## Architecture

```
uv_app/
├── config.py              # Configuration settings
├── logging_config.py      # Logging setup
├── main.py               # Main entry point
├── person.py             # TrackedPerson class
├── tracker.py            # Main tracking logic
├── utils.py              # Utility functions
├── extras.py             # Body/pose processing
└── modules/
    ├── capture_manager.py # Snapshot management
    └── analyzers/
        ├── base.py        # Base analyzer classes
        ├── emotion_analyzer.py
        └── pose_analyzer.py
```

## Development

### Code Style
The project follows Python best practices:
- Type hints throughout
- Comprehensive docstrings
- Modular architecture
- Error handling and logging

### Adding Features
1. Create new modules in appropriate directories
2. Follow the established patterns for logging and configuration
3. Add comprehensive type hints and docstrings
4. Register new components in the main tracker

## Troubleshooting

### Common Issues

1. **Camera not opening**: Check video source index or path
2. **Poor face recognition**: Ensure good lighting and face visibility
3. **Performance issues**: Adjust `RESIZE_MAX` in config for faster processing
4. **Missing dependencies**: Install all requirements from `requirements.txt`

### Debug Mode
Enable debug logging for detailed information:
```python
from uv_app.config import LOG_LEVEL
LOG_LEVEL = "DEBUG"
```

## License

This project is open source. See the license file for details.