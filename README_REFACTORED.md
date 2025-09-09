# Face Tracker - Refactored Modular Architecture

A highly modular real-time face tracking and recognition system built with Python, OpenCV, and face_recognition.

## 🏗️ New Modular Architecture

The project has been completely refactored into a clean, modular architecture that's easy to understand, maintain, and extend.

### 📁 Project Structure

```
uv_app/
├── __init__.py
├── __main__.py              # Package entry point
├── app.py                   # Main application class
├── main.py                  # Legacy entry point (backward compatibility)
├── config.py                # Configuration settings
│
├── core/                    # Core functionality modules
│   ├── __init__.py
│   ├── detection.py         # Face and body detection
│   ├── recognition.py       # Face recognition and matching
│   ├── tracking.py          # Main tracking coordination
│   ├── person.py            # Person data model
│   └── storage.py           # Data persistence
│
├── ui/                      # User interface modules
│   ├── __init__.py
│   ├── display.py           # Video display and visualization
│   └── management.py        # People management tools
│
├── utils/                   # Utility modules
│   ├── __init__.py
│   ├── helpers.py           # Helper functions
│   └── validators.py        # Data validation
│
└── scripts/                 # Management scripts
    ├── manage_people.py     # People management CLI
    └── cleanup_duplicates.py # Duplicate cleanup CLI
```

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
uv sync

# Or with pip
pip install -r requirements.txt
```

### Running the Application

```bash
# New modular way (recommended)
python -m uv_app

# With options
python -m uv_app --help
python -m uv_app --source 0 --enable-body --enable-pose

# Legacy way (still works)
python uv_app/main.py
```

### Management Tools

```bash
# List all tracked people
python uv_app/scripts/manage_people.py --list

# View specific person details
python uv_app/scripts/manage_people.py --view 1

# Rename a person
python uv_app/scripts/manage_people.py --rename 1 "John Doe"

# Find and merge duplicates
python uv_app/scripts/cleanup_duplicates.py --interactive
```

## 🔧 Core Modules

### Detection (`core/detection.py`)
- **FaceDetector**: Handles face detection and encoding extraction
- **BodyDetector**: Manages body detection and pose estimation using MediaPipe

### Recognition (`core/recognition.py`)
- **FaceRecognizer**: Manages face recognition, matching, and person creation
- Handles candidate face processing and duplicate prevention
- Manages lost people re-detection

### Tracking (`core/tracking.py`)
- **PersonTracker**: Main coordination class that orchestrates all components
- Processes video frames and coordinates detection, recognition, and display

### Person (`core/person.py`)
- **TrackedPerson**: Data model for tracked individuals
- Handles face data, emotions, actions, and persistence
- Manages face image storage and metadata

### Storage (`core/storage.py`)
- **PersonStorage**: Handles data persistence and retrieval
- Manages person directories and file operations
- Provides cleanup and maintenance functions

## 🎨 UI Modules

### Display (`ui/display.py`)
- **FaceDisplayManager**: Handles face bounding box drawing and labeling
- **VideoDisplayManager**: Manages video display and window operations

### Management (`ui/management.py`)
- **PeopleManager**: CLI and programmatic people management
- **DuplicateCleaner**: Finds and merges duplicate people

## 🛠️ Utility Modules

### Helpers (`utils/helpers.py`)
- Frame processing and resizing utilities
- Face encoding calculations and validations
- File and data manipulation helpers

### Validators (`utils/validators.py`)
- **ConfigValidator**: Configuration parameter validation
- **ImageValidator**: Image and frame validation
- **PersonValidator**: Person data validation
- **FileValidator**: File operation validation

## 📊 Key Features

### Enhanced Modularity
- **Separation of Concerns**: Each module has a single, clear responsibility
- **Loose Coupling**: Modules interact through well-defined interfaces
- **High Cohesion**: Related functionality is grouped together

### Improved Maintainability
- **Clear Structure**: Easy to find and modify specific functionality
- **Type Hints**: Full type annotations for better IDE support
- **Documentation**: Comprehensive docstrings for all classes and methods
- **Validation**: Built-in data validation and error handling

### Better Extensibility
- **Plugin Architecture**: Easy to add new detection or recognition methods
- **Configurable**: All settings centralized in config.py
- **Modular UI**: Separate display and management interfaces

### Enhanced Management
- **CLI Tools**: Command-line interfaces for common tasks
- **Data Export**: Export person data to JSON
- **Duplicate Detection**: Automatic finding and merging of duplicate people
- **Cleanup Tools**: Remove empty or invalid data

## 🔄 Migration from Old Structure

The refactored code maintains backward compatibility:

```python
# Old way (still works)
from uv_app.tracker import run_tracker
run_tracker(video_source=0, enable_face=True)

# New way (recommended)
from uv_app.app import TrackerApp
app = TrackerApp(enable_face=True)
app.run(video_source=0)
```

## 🎯 Usage Examples

### Basic Face Tracking
```python
from uv_app.app import TrackerApp

app = TrackerApp(enable_face=True)
app.run(video_source=0)  # Webcam
```

### Advanced Tracking with Body and Pose
```python
from uv_app.app import TrackerApp

app = TrackerApp(enable_face=True, enable_body=True, enable_pose=True)
app.run(video_source="path/to/video.mp4")
```

### Programmatic People Management
```python
from uv_app.ui.management import PeopleManager

manager = PeopleManager()
people = manager.list_people()
manager.rename_person(1, "John Doe")
```

### Custom Detection Pipeline
```python
from uv_app.core.detection import FaceDetector
from uv_app.core.recognition import FaceRecognizer

detector = FaceDetector()
recognizer = FaceRecognizer()

# Process frame
locations, encodings = detector.detect_faces(frame)
for encoding, location in zip(encodings, locations):
    person = recognizer.process_face(encoding, face_img, location, frame)
```

## 🧪 Testing and Validation

The refactored code includes comprehensive validation:

```python
from uv_app.utils.validators import ConfigValidator, ImageValidator

# Validate configuration
ConfigValidator.validate_threshold(0.6)  # True
ConfigValidator.validate_positive_int(5)  # True

# Validate image data
ImageValidator.validate_frame(frame)  # True
ImageValidator.validate_face_encoding(encoding)  # True
```

## 📈 Performance Improvements

- **Modular Loading**: Only load required components
- **Efficient Processing**: Optimized face detection pipeline
- **Memory Management**: Better handling of face encodings and images
- **Validation**: Early validation prevents processing errors

## 🔧 Configuration

All settings are centralized in `config.py`:

```python
# Detection settings
MATCH_THRESHOLD = 0.6
CANDIDATE_THRESHOLD = 0.7
MIN_FRAMES_TO_CONFIRM = 5

# Storage settings
SAVE_DIR = "tracked_people"
MAX_FACE_IMAGES = 50
MAX_MISSED_FRAMES = 50

# Processing settings
RESIZE_MAX = 640
```

## 🚀 Future Extensions

The modular architecture makes it easy to add:

- **New Detection Methods**: Add new face/body detection algorithms
- **Custom Recognition**: Implement different face recognition backends
- **Additional UI**: Web interface, mobile app, etc.
- **Data Analytics**: Person behavior analysis and reporting
- **Integration**: Connect with external systems and databases

## 📝 Development Guidelines

1. **Single Responsibility**: Each module should have one clear purpose
2. **Type Hints**: Always include type annotations
3. **Documentation**: Write comprehensive docstrings
4. **Validation**: Validate all inputs and data
5. **Testing**: Test new functionality thoroughly
6. **Error Handling**: Handle errors gracefully with informative messages

This refactored architecture provides a solid foundation for building advanced face tracking and recognition applications while maintaining simplicity and ease of use.
