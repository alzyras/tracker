# uv_app/demo_framework.py

"""
Demo script showing how to use the new robust tracking framework.
This script demonstrates:
1. The enhanced logging system
2. The data management system
3. The modular plugin system
4. Configuration-based control of features
"""

import time
import numpy as np
import cv2
from typing import List

from .core.person import TrackedPerson
from .core.data_manager import DataManager
from .plugins import PluginManager, PLUGIN_REGISTRY
from .config import PLUGIN_CONFIG
from .core.logging import get_logger

logger = get_logger()


def demo_framework():
    """Demonstrate the new framework capabilities."""
    logger.info("🚀 Starting framework demonstration")
    
    # Initialize data manager
    data_manager = DataManager()
    
    # Initialize plugin manager
    plugin_manager = PluginManager()
    
    # Register plugins based on configuration
    logger.info("🔌 Initializing plugins...")
    
    # Register API emotion plugin if enabled
    if PLUGIN_CONFIG.get('api_emotion_plugin_enabled', False):
        api_plugin_cls = PLUGIN_REGISTRY.get("api_emotion")
        if api_plugin_cls:
            api_url = PLUGIN_CONFIG.get('api_emotion_api_url', 'http://localhost:8080')
            api_plugin = api_plugin_cls(
                api_url=api_url,
                update_interval_ms=PLUGIN_CONFIG.get('api_emotion_plugin_interval', 2000)
            )
            plugin_manager.register_plugin(api_plugin)
    
    # Register local emotion plugin if enabled
    if PLUGIN_CONFIG.get('emotion_plugin_enabled', False):
        emotion_plugin_cls = PLUGIN_REGISTRY.get("emotion")
        if emotion_plugin_cls:
            emotion_plugin = emotion_plugin_cls(
                update_interval_ms=PLUGIN_CONFIG.get('emotion_plugin_interval', 2000)
            )
            plugin_manager.register_plugin(emotion_plugin)
    
    # Register simple emotion plugin if enabled
    if PLUGIN_CONFIG.get('simple_emotion_enabled', False):
        simple_emotion_plugin_cls = PLUGIN_REGISTRY.get("simple_emotion")
        if simple_emotion_plugin_cls:
            simple_emotion_plugin = simple_emotion_plugin_cls(
                update_interval_ms=PLUGIN_CONFIG.get('emotion_plugin_interval', 2000)
            )
            plugin_manager.register_plugin(simple_emotion_plugin)
    
    # Register activity plugin if enabled
    if PLUGIN_CONFIG.get('activity_plugin_enabled', True):
        activity_plugin_cls = PLUGIN_REGISTRY.get("activity")
        if activity_plugin_cls:
            activity_plugin = activity_plugin_cls(
                update_interval_ms=PLUGIN_CONFIG.get('activity_plugin_interval', 5000)
            )
            plugin_manager.register_plugin(activity_plugin)
    
    # Register emotion logger plugin if enabled
    if PLUGIN_CONFIG.get('emotion_logger_enabled', True):
        emotion_logger_cls = PLUGIN_REGISTRY.get("emotion_logger")
        if emotion_logger_cls:
            emotion_logger = emotion_logger_cls(
                update_interval_ms=PLUGIN_CONFIG.get('emotion_logger_interval', 5000)
            )
            plugin_manager.register_plugin(emotion_logger)
    
    # Register person event logger plugin if enabled
    if PLUGIN_CONFIG.get('person_event_logger_enabled', True):
        person_event_logger_cls = PLUGIN_REGISTRY.get("person_event_logger")
        if person_event_logger_cls:
            person_event_logger = person_event_logger_cls(
                update_interval_ms=PLUGIN_CONFIG.get('person_event_logger_interval', 1000)
            )
            plugin_manager.register_plugin(person_event_logger)
    
    # Create some demo persons using the original TrackedPerson class
    logger.info("👤 Creating demo persons...")
    person1 = TrackedPerson(1)
    person2 = TrackedPerson(2)
    
    # Set names
    person1.set_name("Agne")
    person2.set_name("Tomas")
    
    # Simulate some face data
    logger.info("📸 Adding face data...")
    dummy_face_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    dummy_encoding = np.random.rand(128)
    
    person1.add_face_data(dummy_encoding, dummy_face_img, (10, 20, 30, 40))
    person2.add_face_data(dummy_encoding, dummy_face_img, (50, 60, 70, 80))
    
    # Simulate tracking updates
    logger.info("🔄 Simulating tracking updates...")
    for i in range(10):
        # Update timestamps
        person1.update()
        person2.update()
        
        # Mark as visible for plugin processing
        person1.is_visible = True
        person2.is_visible = True
        
        # Process with plugins (using dummy frame)
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        plugin_manager.process_people([person1, person2], dummy_frame)
        
        # Add some plugin results manually for demo
        if i % 3 == 0:  # Every 3rd iteration
            data_manager.add_plugin_result(1, "emotion", {
                "emotion": "happy" if i % 2 == 0 else "neutral",
                "confidence": 0.85
            })
            
            data_manager.add_plugin_result(2, "emotion", {
                "emotion": "focused" if i % 2 == 0 else "happy",
                "confidence": 0.92
            })
        
        time.sleep(0.5)
    
    # Show results
    logger.info("📊 Showing results...")
    # Create a simple summary for the TrackedPerson objects
    summary = {
        "total_persons": 2,
        "persons": {
            1: {
                "name": person1.name,
                "face_count": person1.get_face_count()
            },
            2: {
                "name": person2.name,
                "face_count": person2.get_face_count()
            }
        }
    }
    logger.info(f"Data Summary: {summary}")
    
    # Show plugin results
    all_results = plugin_manager.get_all_results()
    logger.info(f"Plugin Results Count: {len(all_results)}")
    
    # Save data
    logger.info("💾 Saving data...")
    person1.save_data()
    person2.save_data()
    
    logger.info("✅ Framework demonstration completed")


if __name__ == "__main__":
    demo_framework()