# UV App - Enhanced Tracking Framework

This document explains the new robust and modular tracking framework that has been implemented.

## Features

1. **Enhanced Logging System**
   - Configurable log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
   - File and console output
   - Structured logging for different types of events
   - Configurable what gets logged

2. **Comprehensive Data Management**
   - Robust storage for person information
   - Face bounding boxes and images
   - Body data and pose information
   - Plugin results storage
   - Persistence to disk

3. **Modular Plugin System**
   - Face-based plugins (emotion detection, face analysis)
   - Body-based plugins (body analysis)
   - Pose-based plugins (pose analysis)
   - Activity tracking plugins
   - API-based plugins (external service integration)
   - Easy to extend with new plugins

4. **Configuration-Based Control**
   - Control what gets logged
   - Enable/disable plugins
   - Configure plugin update intervals
   - Modular configuration system

## Logging System

The new logging system provides configurable output to both console and files:

```python
from uv_app.core.logging import get_logger

logger = get_logger()

# Different log levels
logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
logger.critical("Critical message")

# Specialized logging functions
logger.log_person_match("Agne", 0.334)
logger.log_detection("faces", 2)
logger.log_tracking_event("Person Found", {"id": 1, "name": "Agne"})
logger.log_plugin_result("emotion", 1, {"emotion": "happy", "confidence": 0.85})
```

### Configuration

Logging can be configured in `config.py`:

```python
LOGGING_CONFIG = {
    'log_level': 'INFO',
    'log_matches': True,
    'log_detections': True,
    'log_tracking_events': True,
    'log_plugin_results': True,
    'enable_file_logging': True,
}
```

## Data Management System

The data management system provides a robust way to store and retrieve all information about tracked persons:

```python
from uv_app.core.data_manager import DataManager

# Initialize data manager
data_manager = DataManager()

# Create a person
person = data_manager.create_person(1)

# Add data
person.add_face_data(encoding, face_image, bbox)
person.add_body_data(body_bbox, pose_landmarks)
person.set_name("Agne")

# Add plugin results
data_manager.add_plugin_result(1, "emotion", {"emotion": "happy", "confidence": 0.85})

# Save to disk
data_manager.save_person_data(1)

# Get summary
summary = data_manager.get_summary()
```

## Plugin System

The plugin system is modular and extensible:

### Available Plugins

1. **EmotionPlugin** - Detects emotions from face images (requires DeepFace)
2. **SimpleEmotionPlugin** - Simple emotion detection based on image analysis
3. **APIEmotionPlugin** - Uses external API for emotion detection (your emotion detection service)
4. **ActivityPlugin** - Tracks person activity over time
5. **FaceImagePlugin** - Analyzes face images for detailed information
6. **BodyAnalysisPlugin** - Analyzes body images for descriptive information
7. **PoseAnalysisPlugin** - Analyzes pose landmarks for descriptive information

### Using Plugins

```python
from uv_app.plugins import PluginManager, APIEmotionPlugin

# Initialize plugin manager
plugin_manager = PluginManager()

# Register API emotion plugin
api_emotion_plugin = APIEmotionPlugin(api_url="http://localhost:8080", update_interval_ms=2000)
plugin_manager.register_plugin(api_emotion_plugin)

# Process people
plugin_manager.process_people(people_list, current_frame)

# Get results
results = plugin_manager.get_all_results()
```

### Configuration

Plugins can be configured in `config.py`:

```python
PLUGIN_CONFIG = {
    'api_emotion_plugin_enabled': True,
    'api_emotion_plugin_interval': 2000,  # milliseconds
    'api_emotion_api_url': 'http://localhost:8080',  # Emotion API URL
    'activity_plugin_enabled': True,
    'activity_plugin_interval': 5000,  # milliseconds
}
```

### API Emotion Plugin

The API emotion plugin integrates with your emotion detection service. It sends face images to your API and properly parses the response format:

```bash
# Example API usage
curl -s -X POST \\
  -F "file=@/path/to/face.jpg" \\
  http://localhost:8080/detect | jq
```

The plugin handles the API response format:
```json
{
  "faces": [
    {
      "box": null,
      "scores": {
        "neutral": 0.7448026537895203,
        "happiness": 0.02442147023975849,
        "surprise": 0.009107688441872597,
        "sadness": 0.21152910590171814,
        "anger": 0.007830388844013214,
        "disgust": 0.0004626452282536775,
        "fear": 0.0008954873774200678,
        "contempt": 0.00095049396622926
      },
      "top_emotion": "neutral"
    }
  ]
}
```

### Creating New Plugins

To create a new plugin, extend one of the base classes:

```python
from uv_app.plugins.base import FacePlugin

class MyCustomPlugin(FacePlugin):
    def __init__(self, update_interval_ms: int = 1000):
        super().__init__("my_plugin", update_interval_ms)
    
    def process_face(self, face_image: np.ndarray, person) -> Dict[str, Any]:
        # Your custom processing logic here
        return {"result": "custom_processing_done"}
```

## Running the Demo

To see the framework in action, run:

```bash
python -m uv_app.demo_framework
```

This will demonstrate:
- Creating persons and adding data
- Processing with plugins
- Saving data to disk
- Retrieving results

## Integration with Existing Code

The new framework is designed to integrate seamlessly with existing code. The `TrackedPerson` class has been updated to work with the new logging and data management systems, and the plugin manager can be easily integrated into the existing tracking loop.