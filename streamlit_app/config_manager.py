"""
Configuration manager for the Streamlit configurator.
Handles saving and loading configuration settings.
"""

import json
import os
from typing import Dict, Any
from pathlib import Path

class ConfigManager:
    """Manages configuration for the UV App."""
    
    def __init__(self, config_file: str = "uv_app_config.json"):
        """Initialize the configuration manager."""
        # Use a path relative to the project root
        self.config_file = Path(__file__).parent.parent / config_file
        self.default_config = {
            "plugin_settings": {},
            "ui_settings": {
                "display_mode": "streamlit",
                "box_color": "#00FF00",
                "font_size": 12,
                "font_color": "#FFFFFF",
                "border_width": 2,
                "box_style": "Solid",
                "font_family": "Arial",
                "bg_color": "#0E1117",
                "opacity": 0.8,
                "show_fps": True,
                "frame_width": 640,
                "frame_height": 480,
                "show_preview": True,
                "preview_interval": 100
            }
        }
    
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Returns:
            Configuration dictionary
        """
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                # Merge with default config to ensure all keys are present
                merged_config = self._merge_configs(self.default_config, config)
                return merged_config
            except Exception as e:
                print(f"Error loading config: {e}")
                return self.default_config.copy()
        else:
            # Create default config file
            self.save_config(self.default_config)
            return self.default_config.copy()
    
    def save_config(self, config: Dict[str, Any]) -> bool:
        """
        Save configuration to file.
        
        Args:
            config: Configuration dictionary to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False
    
    def _merge_configs(self, default: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge user configuration with default configuration.
        
        Args:
            default: Default configuration
            user: User configuration
            
        Returns:
            Merged configuration
        """
        merged = default.copy()
        
        for key, value in user.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
                
        return merged
    
    def get_plugin_config(self, plugin_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific plugin.
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            Plugin configuration dictionary
        """
        config = self.load_config()
        return config.get("plugin_settings", {}).get(plugin_name, {})
    
    def update_plugin_config(self, plugin_name: str, plugin_config: Dict[str, Any]) -> bool:
        """
        Update configuration for a specific plugin.
        
        Args:
            plugin_name: Name of the plugin
            plugin_config: Plugin configuration to update
            
        Returns:
            True if successful, False otherwise
        """
        config = self.load_config()
        if "plugin_settings" not in config:
            config["plugin_settings"] = {}
        config["plugin_settings"][plugin_name] = plugin_config
        return self.save_config(config)
    
    def get_ui_config(self) -> Dict[str, Any]:
        """
        Get UI configuration.
        
        Returns:
            UI configuration dictionary
        """
        config = self.load_config()
        return config.get("ui_settings", {})
    
    def update_ui_config(self, ui_config: Dict[str, Any]) -> bool:
        """
        Update UI configuration.
        
        Args:
            ui_config: UI configuration to update
            
        Returns:
            True if successful, False otherwise
        """
        config = self.load_config()
        config["ui_settings"] = ui_config
        return self.save_config(config)

# Global config manager instance
config_manager = ConfigManager()