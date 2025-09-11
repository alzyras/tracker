"""
Configuration loader for the UV App.
Loads configuration saved by the Streamlit configurator.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import os

class UVAppConfigLoader:
    """Loads configuration for the UV App from the Streamlit configurator."""
    
    def __init__(self, config_file: str = "uv_app_config.json"):
        """Initialize the configuration loader."""
        # Look for the config file in the project root
        self.config_file = Path(__file__).parent.parent / config_file
        self._config = None
        self._last_mtime = 0.0
    
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Returns:
            Configuration dictionary
        """
        # Reload when file changes on disk
        if self.config_file.exists():
            try:
                mtime = os.path.getmtime(self.config_file)
                if self._config is None or mtime != self._last_mtime:
                    with open(self.config_file, 'r') as f:
                        self._config = json.load(f)
                    self._last_mtime = mtime
                return self._config
            except Exception as e:
                print(f"Error loading config: {e}")
                return {}
        else:
            return {}

    def refresh(self) -> None:
        """Force reload of configuration on next access."""
        self._config = None
    
    def get_plugin_settings(self) -> Dict[str, Any]:
        """
        Get plugin settings from configuration.
        
        Returns:
            Plugin settings dictionary
        """
        config = self.load_config()
        return config.get("plugin_settings", {})
    
    def get_ui_settings(self) -> Dict[str, Any]:
        """
        Get UI settings from configuration.
        
        Returns:
            UI settings dictionary
        """
        config = self.load_config()
        return config.get("ui_settings", {})
    
    def is_plugin_enabled(self, plugin_name: str) -> bool:
        """
        Check if a plugin is enabled.
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            True if plugin is enabled, False otherwise
        """
        plugin_settings = self.get_plugin_settings()
        plugin_config = plugin_settings.get(plugin_name, {})
        return plugin_config.get("enabled", False)
    
    def get_plugin_parameter(self, plugin_name: str, parameter_name: str, default=None):
        """
        Get a specific parameter for a plugin.
        
        Args:
            plugin_name: Name of the plugin
            parameter_name: Name of the parameter
            default: Default value if parameter not found
            
        Returns:
            Parameter value or default
        """
        plugin_settings = self.get_plugin_settings()
        plugin_config = plugin_settings.get(plugin_name, {})
        return plugin_config.get(parameter_name, default)
    
    def get_ui_parameter(self, parameter_name: str, default=None):
        """
        Get a specific UI parameter.
        
        Args:
            parameter_name: Name of the parameter
            default: Default value if parameter not found
            
        Returns:
            Parameter value or default
        """
        ui_settings = self.get_ui_settings()
        return ui_settings.get(parameter_name, default)

# Global config loader instance
config_loader = UVAppConfigLoader()