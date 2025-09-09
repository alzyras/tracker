# Face Tracking Framework with Plugin System

A clean, modular face tracking framework that provides raw person detection and tracking capabilities with an extensible plugin system for adding custom analysis features.

## 🎯 **Core Philosophy**

This framework is designed to be:
- **Raw and Clean**: Only essential tracking functionality, no built-in analysis
- **Extensible**: Plugin system for adding custom features
- **Real-time**: Access to current face/body/pose images and coordinates
- **Modular**: Easy to integrate into larger applications

## 🏗️ **Architecture**

### Core Components

```
uv_app/
├── core/                    # Core tracking functionality
│   ├── detection.py         # Face and body detection
│   ├── recognition.py       # Face recognition and matching
│   ├── tracking.py          # Main tracking coordination
│   ├── person.py            # Person data model
│   └── storage.py           # Data persistence
├── plugins/                 # Plugin system
│   ├── base.py              # Base plugin classes
│   ├── manager.py           # Plugin management
│   ├── emotion_plugin.py    # Example emotion plugins
│   ├── activity_plugin.py   # Example activity plugins
│   └── api_plugin.py        # Generic API plugins
└── examples/                # Example applications
    └── plugin_example.py    # Plugin system demo
```

## 🚀 **Quick Start**

### Basic Usage

```python
from uv_app.core.tracking import PersonTracker

# Create tracker
tracker = PersonTracker(
    enable_face=True,
    enable_body=True,
    enable_pose=True
)

# Process frames
while True:
    ret, frame = cap.read()
    processed_frame = tracker.process_frame(frame)
    
    # Get current data
    visible_people = tracker.get_visible_people()
    for person in visible_people:
        face_image = person.get_current_face_image()
        body_image = person.get_current_body_image()
        coordinates = person.get_current_coordinates()
```

### With Plugins

```python
from uv_app.core.tracking import PersonTracker
from uv_app.plugins.emotion_plugin import SimpleEmotionPlugin
from uv_app.plugins.activity_plugin import SimpleActivityPlugin

# Create tracker
tracker = PersonTracker(enable_face=True, enable_body=True, enable_pose=True)

# Register plugins
emotion_plugin = SimpleEmotionPlugin(update_interval_ms=1000)
activity_plugin = SimpleActivityPlugin(update_interval_ms=2000)

tracker.register_plugin(emotion_plugin)
tracker.register_plugin(activity_plugin)

# Process frames
while True:
    ret, frame = cap.read()
    processed_frame = tracker.process_frame(frame)
    
    # Get plugin results
    results = tracker.get_plugin_results()
    print(results)
```

## 📊 **Person Data Model**

### Essential Data Only

```python
class TrackedPerson:
    # Core tracking data
    track_id: int
    name: Optional[str]
    face_encodings: List[np.ndarray]
    face_images: List[np.ndarray]
    face_boxes: List[Tuple]
    body_boxes: List[Tuple]
    pose_landmarks: List[Any]
    
    # Current state (updated each frame)
    current_face_image: Optional[np.ndarray]
    current_body_image: Optional[np.ndarray]
    current_pose_landmarks: Optional[Any]
    current_face_bbox: Optional[Tuple]
    current_body_bbox: Optional[Tuple]
    is_visible: bool
```

### Real-time Data Access

```python
# Get current images
face_img = person.get_current_face_image()
body_img = person.get_current_body_image()
pose_landmarks = person.get_current_pose_landmarks()

# Get coordinates
coordinates = person.get_current_coordinates()
face_bbox = person.get_current_face_bbox()
body_bbox = person.get_current_body_bbox()

# Get stored data
all_faces = person.get_all_face_images()
face_count = person.get_face_count()
```

## 🔌 **Plugin System**

### Plugin Types

1. **FacePlugin**: Processes face images
2. **BodyPlugin**: Processes body images  
3. **PosePlugin**: Processes pose landmarks
4. **BasePlugin**: Custom plugin base class

### Creating a Plugin

```python
from uv_app.plugins.base import FacePlugin

class MyEmotionPlugin(FacePlugin):
    def __init__(self):
        super().__init__("my_emotion", update_interval_ms=1000)
    
    def process_face(self, face_image: np.ndarray, person) -> Dict[str, Any]:
        # Your emotion detection logic here
        emotion = self.detect_emotion(face_image)
        return {
            "emotion": emotion,
            "confidence": 0.8
        }
    
    def detect_emotion(self, face_image):
        # Implement your detection logic
        return "happy"
```

### API Plugin Example

```python
from uv_app.plugins.api_plugin import create_emotion_api_plugin

# Create API plugin
emotion_api = create_emotion_api_plugin(
    api_url="https://your-api.com/emotion",
    api_key="your-api-key"
)

tracker.register_plugin(emotion_api)
```

## 📋 **Available Methods**

### Tracker Methods

```python
# Core tracking
tracker.process_frame(frame)                    # Process a frame
tracker.get_visible_people()                    # Get currently visible people
tracker.get_all_people()                        # Get all tracked people
tracker.get_person_by_id(track_id)              # Get specific person

# Plugin management
tracker.register_plugin(plugin)                 # Register a plugin
tracker.get_plugin_results()                    # Get all plugin results
tracker.get_plugin_results(person_id=1)         # Get results for person
tracker.get_plugin_results(plugin_name="emotion") # Get results for plugin

# Data management
tracker.save_all_data()                         # Save all data
```

### Person Methods

```python
# Current state
person.get_current_face_image()                 # Current face image
person.get_current_body_image()                 # Current body image
person.get_current_pose_landmarks()             # Current pose landmarks
person.get_current_face_bbox()                  # Current face bounding box
person.get_current_body_bbox()                  # Current body bounding box
person.get_current_coordinates()                # All current coordinates

# Stored data
person.get_all_face_images()                    # All stored face images
person.get_face_count()                         # Number of stored faces
person.set_name("John Doe")                     # Set person name

# State management
person.is_visible                               # Whether person is currently visible
person.mark_not_visible()                       # Mark as not visible
```

## 🎨 **Example Applications**

### 1. Basic Tracking

```python
from uv_app.core.tracking import PersonTracker

tracker = PersonTracker(enable_face=True)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    processed_frame = tracker.process_frame(frame)
    
    # Show current people
    visible_people = tracker.get_visible_people()
    print(f"Visible people: {len(visible_people)}")
    
    cv2.imshow("Tracker", processed_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

### 2. With Emotion Detection

```python
from uv_app.core.tracking import PersonTracker
from uv_app.plugins.emotion_plugin import SimpleEmotionPlugin

tracker = PersonTracker(enable_face=True)
emotion_plugin = SimpleEmotionPlugin(update_interval_ms=1000)
tracker.register_plugin(emotion_plugin)

while True:
    ret, frame = cap.read()
    processed_frame = tracker.process_frame(frame)
    
    # Get emotion results
    results = tracker.get_plugin_results(plugin_name="simple_emotion")
    for person_id, result in results.items():
        print(f"Person {person_id}: {result['emotion']}")
```

### 3. Custom Analysis

```python
from uv_app.core.tracking import PersonTracker
from uv_app.plugins.base import FacePlugin

class CustomAnalysisPlugin(FacePlugin):
    def __init__(self):
        super().__init__("custom_analysis", update_interval_ms=2000)
    
    def process_face(self, face_image, person):
        # Your custom analysis here
        analysis_result = self.analyze_face(face_image)
        return {"custom_result": analysis_result}

tracker = PersonTracker(enable_face=True)
custom_plugin = CustomAnalysisPlugin()
tracker.register_plugin(custom_plugin)
```

## 🔧 **Configuration**

All settings in `config.py`:

```python
# Tracking settings
MATCH_THRESHOLD = 0.6          # Face recognition threshold
CANDIDATE_THRESHOLD = 0.7      # New person threshold
MIN_FRAMES_TO_CONFIRM = 5      # Frames to confirm new person
MAX_MISSED_FRAMES = 50         # Frames before marking as lost
MAX_FACE_IMAGES = 30           # Max face images per person

# Processing settings
RESIZE_MAX = 640               # Max frame size
VERBOSE_TRACKING = True        # Show tracking messages
```

## 🚀 **Running Examples**

```bash
# Basic tracking
python -m uv_app

# Plugin example
python uv_app/examples/plugin_example.py

# With specific features
python -m uv_app --enable-body --enable-pose
```

## 🔌 **Plugin Development**

### Plugin Lifecycle

1. **Initialization**: Plugin is created and registered
2. **Processing**: `process_person()` called at specified intervals
3. **Results**: Results stored and accessible via `get_plugin_results()`

### Best Practices

1. **Error Handling**: Always wrap processing in try-catch
2. **Performance**: Use appropriate update intervals
3. **Resource Management**: Clean up resources in plugin
4. **Thread Safety**: Plugins run in main thread

### Example Plugin Template

```python
from uv_app.plugins.base import FacePlugin

class MyPlugin(FacePlugin):
    def __init__(self):
        super().__init__("my_plugin", update_interval_ms=1000)
        # Initialize your resources
    
    def process_face(self, face_image, person):
        try:
            # Your processing logic
            result = self.process(face_image)
            return {"result": result}
        except Exception as e:
            return {"error": str(e)}
    
    def process(self, face_image):
        # Implement your logic
        pass
```

## 🎯 **Use Cases**

- **Security Systems**: Person detection and tracking
- **Analytics**: Behavior analysis with custom plugins
- **Access Control**: Face recognition for authentication
- **Research**: Data collection for ML training
- **Monitoring**: Real-time person monitoring
- **Integration**: Embed in larger applications

This framework provides a solid foundation for any person tracking application while keeping the core clean and extensible through the plugin system.
