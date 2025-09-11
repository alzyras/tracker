"""
Demo script showing how configuration from Streamlit is applied to the UV App.
"""

import sys
from pathlib import Path

# Add the project root to sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from uv_app.config_loader import config_loader

def demo_configuration_application():
    """Demonstrate how configuration from Streamlit is applied to the UV App."""
    print("UV App Configuration Demo")
    print("=" * 30)
    
    # Load the configuration that would be saved by Streamlit
    print("1. Loading configuration from Streamlit...")
    config = config_loader.load_config()
    
    if not config:
        print("   No configuration found. Using defaults.")
        return
    
    print("   Configuration loaded successfully!")
    
    # Show plugin configuration
    print("\n2. Plugin Configuration:")
    plugin_settings = config_loader.get_plugin_settings()
    for plugin_name, settings in plugin_settings.items():
        enabled = settings.get('enabled', False)
        print(f"   {plugin_name}: {'ENABLED' if enabled else 'DISABLED'}")
        if enabled:
            # Show plugin-specific settings
            for key, value in settings.items():
                if key != 'enabled':
                    print(f"     {key}: {value}")
    
    # Show UI configuration
    print("\n3. UI Configuration:")
    ui_settings = config_loader.get_ui_settings()
    display_mode = ui_settings.get('display_mode', 'opencv')
    print(f"   Display Mode: {display_mode}")
    print(f"   Box Color: {ui_settings.get('box_color', '#00FF00')}")
    print(f"   Font Size: {ui_settings.get('font_size', 12)}")
    print(f"   Border Width: {ui_settings.get('border_width', 2)}")
    
    # Show how this would be applied in the main app
    print("\n4. How this configuration is applied in the main app:")
    print("   - Plugins are registered based on their 'enabled' status")
    print("   - Plugin parameters (update intervals, API URLs, etc.) are passed to plugin constructors")
    print("   - UI settings control how tracking results are displayed")
    print("   - Display mode determines whether to show results in OpenCV window, Streamlit, or neither")

if __name__ == "__main__":
    demo_configuration_application()