# uv_app/plugins/manager.py

import time
from typing import List, Dict, Any
import numpy as np
from .base import BasePlugin
from ..core.logging import get_logger

logger = get_logger()


class PluginManager:
    """Manages and executes tracking plugins."""
    
    def __init__(self):
        self.plugins: List[BasePlugin] = []
        self.results: Dict[str, Dict[str, Any]] = {}
        logger.debug("Initialized PluginManager")
    
    def register_plugin(self, plugin: BasePlugin) -> None:
        """Register a plugin."""
        self.plugins.append(plugin)
        logger.info(f"Registered plugin: {plugin.name}")
    
    def unregister_plugin(self, plugin_name: str) -> None:
        """Unregister a plugin by name."""
        self.plugins = [p for p in self.plugins if p.name != plugin_name]
        if plugin_name in self.results:
            del self.results[plugin_name]
        logger.info(f"Unregistered plugin: {plugin_name}")
    
    def process_people(self, people: List, frame: np.ndarray) -> None:
        """Process all people with all plugins."""
        current_time_ms = int(time.time() * 1000)
        
        for person in people:
            if not person.is_visible:
                continue
            
            for plugin in self.plugins:
                # Run emotion plugins every frame to log emotions continuously
                is_emotion_plugin = plugin.name in ("api_emotion", "emotion", "simple_emotion")
                if is_emotion_plugin or plugin.should_update(current_time_ms):
                    try:
                        result = plugin.process_person(person, frame)
                        self.results[f"{plugin.name}_{person.track_id}"] = {
                            "person_id": person.track_id,
                            "plugin": plugin.name,
                            "result": result,
                            "timestamp": current_time_ms
                        }
                        plugin.update_timestamp(current_time_ms)
                        # Always log emotions immediately when identified
                        if plugin.name in ("api_emotion", "emotion", "simple_emotion") and isinstance(result, dict):
                            emotion = result.get("emotion")
                            confidence = result.get("confidence")
                            if emotion:
                                # Confidence might be dict for API; extract best if needed
                                if isinstance(confidence, dict):
                                    confidence = confidence.get(emotion, 0.0)
                                try:
                                    conf_val = float(confidence) if confidence is not None else 0.0
                                except Exception:
                                    conf_val = 0.0
                                person_name = person.name if person.name else f"Person ID {person.track_id}"
                                logger.info(f"😊 {person_name}: {emotion} ({conf_val:.2f})")
                        # Respect config for generic plugin result logging
                        logger.log_plugin_result(plugin.name, person.track_id, result)
                    except Exception as e:
                        error_msg = f"Error in plugin {plugin.name}: {e}"
                        logger.error(error_msg)
                        self.results[f"{plugin.name}_{person.track_id}"] = {
                            "person_id": person.track_id,
                            "plugin": plugin.name,
                            "error": str(e),
                            "timestamp": current_time_ms
                        }
    
    def get_results_for_person(self, person_id: int) -> Dict[str, Any]:
        """Get all results for a specific person."""
        person_results = {}
        for key, result in self.results.items():
            if result["person_id"] == person_id:
                plugin_name = result["plugin"]
                person_results[plugin_name] = result["result"]
        return person_results
    
    def get_results_for_plugin(self, plugin_name: str) -> Dict[int, Any]:
        """Get all results for a specific plugin."""
        plugin_results = {}
        for key, result in self.results.items():
            if result["plugin"] == plugin_name:
                person_id = result["person_id"]
                plugin_results[person_id] = result["result"]
        return plugin_results
    
    def get_all_results(self) -> Dict[str, Dict[str, Any]]:
        """Get all results."""
        return self.results.copy()
    
    def clear_old_results(self, max_age_ms: int = 30000) -> None:
        """Clear results older than max_age_ms."""
        current_time_ms = int(time.time() * 1000)
        old_count = len(self.results)
        self.results = {
            key: result for key, result in self.results.items()
            if current_time_ms - result["timestamp"] < max_age_ms
        }
        cleared_count = old_count - len(self.results)
        if cleared_count > 0:
            logger.debug(f"Cleared {cleared_count} old plugin results")
    
    def enable_plugin(self, plugin_name: str) -> None:
        """Enable a plugin."""
        for plugin in self.plugins:
            if plugin.name == plugin_name:
                plugin.enable()
                logger.info(f"Enabled plugin: {plugin_name}")
                break
    
    def disable_plugin(self, plugin_name: str) -> None:
        """Disable a plugin."""
        for plugin in self.plugins:
            if plugin.name == plugin_name:
                plugin.disable()
                logger.info(f"Disabled plugin: {plugin_name}")
                break
    
    def get_plugin_status(self) -> Dict[str, bool]:
        """Get status of all plugins."""
        status = {plugin.name: plugin.enabled for plugin in self.plugins}
        logger.debug(f"Plugin status: {status}")
        return status
