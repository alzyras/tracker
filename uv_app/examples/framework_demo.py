# uv_app/examples/framework_demo.py

"""
Demo application showing how to use the new robust tracking framework.
This example demonstrates:
1. Comprehensive logging with configurable levels
2. Data management for all person information
3. Modular plugin system with API-based extensions
"""

import cv2
import numpy as np
from typing import List
import time

from ..core.logging import get_logger, setup_logging
from ..core.data_manager import DataManager
from ..core.detection import FaceDetector, BodyDetector
from ..plugins import PluginManager, PLUGIN_REGISTRY
from ..config import LOGGING_CONFIG, PLUGIN_CONFIG

# Setup logging
logger = setup_logging(LOGGING_CONFIG)

def create_demo_plugins() -> PluginManager:
    """Create and configure plugins based on configuration."""
    plugin_manager = PluginManager()
    
    # Register plugins based on configuration
    if PLUGIN_CONFIG.get('emotion_plugin_enabled', True):
        emotion_plugin = PLUGIN_REGISTRY["emotion"](
            update_interval_ms=PLUGIN_CONFIG.get('emotion_plugin_interval', 2000)
        )
        plugin_manager.register_plugin(emotion_plugin)
    
    if PLUGIN_CONFIG.get('activity_plugin_enabled', True):
        activity_plugin = PLUGIN_REGISTRY["activity"](
            update_interval_ms=PLUGIN_CONFIG.get('activity_plugin_interval', 5000)
        )
        plugin_manager.register_plugin(activity_plugin)
    
    # Register additional plugins
    face_image_plugin = PLUGIN_REGISTRY["face_image"](update_interval_ms=1000)
    plugin_manager.register_plugin(face_image_plugin)
    
    body_analysis_plugin = PLUGIN_REGISTRY["body_analysis"](update_interval_ms=2000)
    plugin_manager.register_plugin(body_analysis_plugin)
    
    pose_analysis_plugin = PLUGIN_REGISTRY["pose_analysis"](update_interval_ms=1500)
    plugin_manager.register_plugin(pose_analysis_plugin)
    
    return plugin_manager

def demo_tracking_framework():
    """Demonstrate the tracking framework capabilities."""
    logger.info("Starting tracking framework demo")
    
    # Initialize components
    data_manager = DataManager()
    face_detector = FaceDetector()
    body_detector = BodyDetector()
    plugin_manager = create_demo_plugins()
    
    logger.info("Framework components initialized")
    
    # Create a few demo persons
    logger.info("Creating demo persons")
    person1 = data_manager.create_person(1)
    person1.set_name("Agne")
    
    person2 = data_manager.create_person(2)
    person2.set_name("Tomas")
    
    # Simulate some tracking data
    logger.info("Simulating tracking data")
    
    # Add some face data to person1
    dummy_face_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    dummy_encoding = np.random.rand(128)
    person1.add_face_data(dummy_encoding, dummy_face_img, (10, 20, 30, 40))
    
    # Add some body data to person2
    person2.add_body_data((50, 60, 150, 200), None, None)
    
    # Simulate some plugin results
    emotion_result = {
        "emotion": "happy",
        "confidence": {"happy": 0.85},
        "method": "deepface"
    }
    data_manager.add_plugin_result(1, "emotion", emotion_result)
    
    activity_result = {
        "duration_seconds": 120.5,
        "frame_count": 300,
        "visible_frames": 280,
        "visibility_ratio": 0.93,
        "movement_score": 0.75,
        "position_count": 45
    }
    data_manager.add_plugin_result(2, "activity", activity_result)
    
    # Process with plugins (mock frame)
    mock_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Set current state for persons
    person1.current_face_image = dummy_face_img
    person1.current_face_bbox = (10, 20, 30, 40)
    person1.is_visible = True
    
    person2.current_body_image = mock_frame
    person2.current_body_bbox = (50, 60, 150, 200)
    person2.is_visible = True
    
    # Process with plugins
    logger.info("Processing persons with plugins")
    plugin_manager.process_people([person1, person2], mock_frame)
    
    # Show results
    logger.info("Demo results:")
    summary = data_manager.get_summary()
    logger.info(f"Tracking summary: {summary}")
    
    # Show plugin results
    all_results = plugin_manager.get_all_results()
    logger.info(f"Plugin results: {len(all_results)} results processed")
    
    # Save data
    logger.info("Saving person data")
    data_manager.save_all_data()
    
    logger.info("Demo completed successfully")

if __name__ == "__main__":
    demo_tracking_framework()