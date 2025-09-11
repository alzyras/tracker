"""
Test script for the UV App configuration loader.
"""

import sys
from pathlib import Path

# Add the project root to sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from uv_app.config_loader import config_loader

def test_config_loader():
    """Test the configuration loader."""
    print("Testing UV App Configuration Loader")
    print("=" * 40)
    
    # Load configuration
    config = config_loader.load_config()
    print("Configuration loaded successfully")
    
    # Get plugin settings
    plugin_settings = config_loader.get_plugin_settings()
    print(f"Plugin settings: {plugin_settings}")
    
    # Get UI settings
    ui_settings = config_loader.get_ui_settings()
    print(f"UI settings: {ui_settings}")
    
    # Test specific plugin
    for plugin_name in ['emotion', 'api_emotion', 'smolvlm_activity']:
        is_enabled = config_loader.is_plugin_enabled(plugin_name)
        print(f"Plugin '{plugin_name}' enabled: {is_enabled}")
        
        if is_enabled:
            update_interval = config_loader.get_plugin_parameter(plugin_name, 'update_interval_ms', 1000)
            print(f"  Update interval: {update_interval}ms")
            
            # For API-based plugins, check API URL
            if 'api' in plugin_name:
                api_url = config_loader.get_plugin_parameter(plugin_name, 'api_url', '')
                print(f"  API URL: {api_url}")
    
    # Test UI parameters
    display_mode = config_loader.get_ui_parameter('display_mode', 'opencv')
    print(f"Display mode: {display_mode}")
    
    box_color = config_loader.get_ui_parameter('box_color', '#00FF00')
    print(f"Box color: {box_color}")

if __name__ == "__main__":
    test_config_loader()