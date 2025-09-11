# Tracker Project

A real-time face, body, and pose tracking application using OpenCV, MediaPipe, and face_recognition.

## Features

- Real-time face detection and recognition
- Full-body bounding box tracking
- Pose detection and tracking
- Webcam and CCTV stream support
- Person re-identification across frames
- **Automatic face saving** - New faces are automatically saved to `tracked_people/` folder
- **ID assignment** - Each person gets a unique ID that persists across sessions
- **Certainty percentage** - Shows match confidence for face recognition
- **Name management** - Assign and display names for recognized people
- **Face management tool** - View, rename, and organize detected faces
- **Enhanced recognition accuracy** - Improved algorithms to prevent misidentification
- **Periodic emotion logging** - Logs person emotions every 5 seconds
- **API emotion integration** - Shows emotions from external emotion detection service
- **Activity recognition with SmolVLM** - Uses SmolVLM API to describe what people are doing

## Installation

### Prerequisites

- Python 3.11 (required)
- Git

### Quick Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd tracker_project
```

2. Install dependencies using uv (recommended):
```bash
# Install uv if you don't have it
pip install uv

# Install project dependencies
uv sync
```

3. Run the application:
```bash
uv run python uv_app/main.py
```

### Alternative Installation with pip

If you prefer using pip instead of uv:

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python uv_app/main.py
```

## Usage

### Running the Tracker

The application will start tracking faces, bodies, and poses from your webcam by default. You can modify the settings in `uv_app/main.py` to:

- Use a different video source (CCTV stream, video file)
- Enable/disable specific tracking features
- Adjust detection parameters

### Example: Using CCTV Stream

Uncomment and modify the CCTV stream section in `main.py`:

```python
run_tracker(
    video_source="http://192.168.1.31:8080/video",
    enable_face=True,
    enable_body=True,
    enable_pose=True
)
```

### Managing Tracked People

Use the built-in management tool to view and organize detected faces:

```bash
uv run python uv_app/manage_people.py
```

This tool allows you to:
- List all tracked people with their IDs and names
- Rename people for easier identification
- View face images for each person

### Cleaning Up Duplicates

If the system creates multiple IDs for the same person, use the cleanup tool:

```bash
uv run python uv_app/cleanup_duplicates.py
```

This tool helps you:
- Find potential duplicate people based on face similarity
- Merge duplicate people into a single ID
- Clean up your tracked people database

### Face Recognition Features

- **New Face Detection**: When a new face is detected, it gets a red bounding box with "NEW ID X" label
- **Recognition**: Recognized faces show green bounding boxes with "ID X (certainty%)" and name if available
- **Automatic Saving**: All face images are automatically saved to `tracked_people/person_X/` folders
- **Persistent IDs**: Person IDs are maintained across application restarts

### Activity Recognition with SmolVLM

The system includes integration with the SmolVLM API for detailed activity recognition. The SmolVLM plugin:

- Captures body images of detected people
- Sends them to the SmolVLM API for natural language descriptions of activities
- Logs what each person is doing in real-time
- Works asynchronously to avoid blocking the main tracking thread

To use the SmolVLM plugin:

1. Start the SmolVLM API server on `http://localhost:9000`
2. Enable the plugin in `uv_app/config.py`:
   ```python
   PLUGIN_CONFIG = {
       # ... other settings
       'smolvlm_plugin_enabled': True,
       'smolvlm_plugin_interval': 5000,  # milliseconds (5 seconds)
       'smolvlm_api_url': 'http://localhost:9000/describe',
   }
   ```
3. Run the application with body detection enabled:
   ```bash
   uv run python uv_app/app.py --enable-body
   ```

Example log output:
```
Person ID 1 is doing: The person is sitting at a desk working on a computer
Person ID 2 is doing: The person is walking across the room
```

### Enhanced Features

For detailed information about the enhanced features, see:
- [Plugin System](README_PLUGIN_SYSTEM.md)

## Configuration

Key configuration parameters can be found in `uv_app/config.py`:

- `MATCH_THRESHOLD`: Face recognition matching threshold (0.45 - stricter than default)
- `CANDIDATE_THRESHOLD`: Minimum confidence for new face candidates (0.4)
- `MAX_MISSED_FRAMES`: Maximum frames a person can be missed before being considered lost (50)
- `MIN_FRAMES_TO_CONFIRM`: Minimum frames required to confirm a new person (30)
- `MAX_FACE_IMAGES`: Maximum face images to store per person (30)
- `SAVE_DIR`: Directory to save tracked people data ("tracked_people")

## File Structure

```
tracked_people/
├── person_1/
│   ├── data.json          # Person metadata (ID, name, etc.)
│   ├── encodings.npy      # Face encodings for recognition
│   ├── face_1.jpg         # Face images
│   └── ...
├── person_2/
│   └── ...
```

## Troubleshooting

### Common Issues

1. **Camera not found**: Make sure your webcam is connected and not being used by another application.

2. **Performance issues**: Try reducing the frame resolution or disabling some tracking features.

3. **Face recognition not working**: Ensure all dependencies are installed correctly:
   ```bash
   uv sync
   ```

4. **No faces being saved**: Check that the `tracked_people` directory exists and is writable.

5. **Multiple IDs for same person**: This can happen with poor lighting or angle changes. Use the cleanup tool to merge duplicates:
   ```bash
   uv run python uv_app/cleanup_duplicates.py
   ```

6. **Recognition accuracy issues**: The system has been enhanced with stricter matching thresholds. If you're still having issues:
   - Clean up person data: `uv run python uv_app/cleanup_person_data.py`
   - Ensure good lighting and clear face visibility
   - Allow time for the system to build accurate profiles

## Dependencies

- OpenCV for computer vision
- face_recognition for face detection and recognition
- MediaPipe for pose and body detection
- NumPy for numerical operations
- dlib for face recognition backend
- Ultralytics YOLO for object detection
- DeepSort for object tracking

## License

[Add your license information here]