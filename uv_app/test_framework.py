#!/usr/bin/env python3
# uv_app/test_framework.py

"""
Test script for the new tracking framework.
This script tests:
1. The enhanced logging system
2. The data management system
3. The modular plugin system
4. Configuration-based control of features
"""

import sys
import os
import numpy as np

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from uv_app.core.logging import get_logger, setup_logging
from uv_app.core.data_manager import DataManager
from uv_app.plugins import PluginManager, PLUGIN_REGISTRY
from uv_app.config import LOGGING_CONFIG, PLUGIN_CONFIG


def test_logging_system():
    """Test the enhanced logging system."""
    print("Testing logging system...")
    
    # Setup logging
    logger = setup_logging(LOGGING_CONFIG)
    
    # Test different log levels
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    
    # Test specialized logging
    logger.log_person_match("Agne", 0.334)
    logger.log_detection("faces", 2)
    logger.log_tracking_event("Person Found", {"id": 1, "name": "Agne"})
    
    print("✅ Logging system test completed")


def test_data_management():
    """Test the data management system."""
    print("Testing data management system...")
    
    # Initialize data manager
    data_manager = DataManager("test_tracked_people")
    
    # Create a person
    person = data_manager.create_person(1)
    person.set_name("Agne")
    
    # Add face data
    dummy_face_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    dummy_encoding = np.random.rand(128)
    person.add_face_data(dummy_encoding, dummy_face_img, (10, 20, 30, 40))
    
    # Add body data
    person.add_body_data((50, 60, 70, 80))
    
    # Add plugin results
    data_manager.add_plugin_result(1, "emotion", {
        "emotion": "happy",
        "confidence": 0.85
    })
    
    # Get summary
    summary = data_manager.get_summary()
    print(f"Data summary: {summary}")
    
    # Save data
    data_manager.save_person_data(1)
    
    print("✅ Data management test completed")


def test_plugin_system():
    """Test the plugin system."""
    print("Testing plugin system...")
    
    # Initialize plugin manager
    plugin_manager = PluginManager()
    
    # Register plugins based on configuration
    if PLUGIN_CONFIG.get('api_emotion_plugin_enabled', False):
        api_plugin_cls = PLUGIN_REGISTRY.get("api_emotion")
        if api_plugin_cls:
            api_url = PLUGIN_CONFIG.get('api_emotion_api_url', 'http://localhost:8080')
            api_plugin = api_plugin_cls(
                api_url=api_url,
                update_interval_ms=PLUGIN_CONFIG.get('api_emotion_plugin_interval', 2000)
            )
            plugin_manager.register_plugin(api_plugin)
            print("✅ Registered API emotion plugin")
    elif PLUGIN_CONFIG.get('emotion_plugin_enabled', True):
        emotion_plugin_cls = PLUGIN_REGISTRY.get("simple_emotion")
        if emotion_plugin_cls:
            emotion_plugin = emotion_plugin_cls(
                update_interval_ms=PLUGIN_CONFIG.get('emotion_plugin_interval', 2000)
            )
            plugin_manager.register_plugin(emotion_plugin)
            print("✅ Registered emotion plugin")
    
    if PLUGIN_CONFIG.get('activity_plugin_enabled', True):
        activity_plugin_cls = PLUGIN_REGISTRY.get("activity")
        if activity_plugin_cls:
            activity_plugin = activity_plugin_cls(
                update_interval_ms=PLUGIN_CONFIG.get('activity_plugin_interval', 5000)
            )
            plugin_manager.register_plugin(activity_plugin)
            print("✅ Registered activity plugin")
    
    # Register emotion logger plugin if enabled
    if PLUGIN_CONFIG.get('emotion_logger_enabled', True):
        emotion_logger_cls = PLUGIN_REGISTRY.get("emotion_logger")
        if emotion_logger_cls:
            emotion_logger = emotion_logger_cls(
                update_interval_ms=PLUGIN_CONFIG.get('emotion_logger_interval', 5000)
            )
            plugin_manager.register_plugin(emotion_logger)
            print("✅ Registered emotion logger plugin")
    
    # Register person event logger plugin if enabled
    if PLUGIN_CONFIG.get('person_event_logger_enabled', True):
        person_event_logger_cls = PLUGIN_REGISTRY.get("person_event_logger")
        if person_event_logger_cls:
            person_event_logger = person_event_logger_cls(
                update_interval_ms=PLUGIN_CONFIG.get('person_event_logger_interval', 1000)
            )
            plugin_manager.register_plugin(person_event_logger)
            print("✅ Registered person event logger plugin")
    
    # Register SmolVLM plugin if enabled
    if PLUGIN_CONFIG.get('smolvlm_plugin_enabled', True):
        smolvlm_plugin_cls = PLUGIN_REGISTRY.get("smolvlm_activity")
        if smolvlm_plugin_cls:
            api_url = PLUGIN_CONFIG.get('smolvlm_api_url', 'http://localhost:9000/describe')
            smolvlm_plugin = smolvlm_plugin_cls(
                api_url=api_url,
                update_interval_ms=PLUGIN_CONFIG.get('smolvlm_plugin_interval', 2000)
            )
            plugin_manager.register_plugin(smolvlm_plugin)
            print("✅ Registered SmolVLM activity plugin")
    
    # Check plugin status
    status = plugin_manager.get_plugin_status()
    print(f"Plugin status: {status}")
    
    print("✅ Plugin system test completed")


def main():
    """Main test function."""
    print("🚀 Testing UV App Framework")
    print("=" * 40)
    
    try:
        test_logging_system()
        print()
        
        test_data_management()
        print()
        
        test_plugin_system()
        print()
        
        print("🎉 All tests completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())