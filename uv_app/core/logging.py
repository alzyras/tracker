# uv_app/core/logging.py

import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional
import json

# Define log levels
LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL,
}

class LoggerManager:
    """Manages the application's logging system with file and console output."""
    
    def __init__(self, name: str = "uv_app", config: Optional[Dict[str, Any]] = None):
        """
        Initialize the logger manager.
        
        Args:
            name: Name of the logger
            config: Configuration dictionary for logging
        """
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(name)

        # Set default log level
        log_level = self.config.get('log_level', 'INFO').upper()
        self.logger.setLevel(LOG_LEVELS.get(log_level, logging.INFO))

        # Always reset handlers to prevent duplication
        self.logger.handlers.clear()
        self._setup_handlers()
    
    def _setup_handlers(self) -> None:
        """Set up console and file handlers based on configuration."""
        # Console handler
        console_handler = logging.StreamHandler()
        console_format = logging.Formatter(
            '%(asctime)s [%(levelname)s] - <%(name)s:%(lineno)d> - %(message)s'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)

        # File handler if enabled
        if self.config.get('enable_file_logging', True):
            # We'll set the log directory when we know the SAVE_DIR
            pass

    def setup_file_logging(self, save_dir: str) -> None:
        """Set up file logging with the given save directory."""
        if self.config.get('enable_file_logging', True):
            log_dir = os.path.join(save_dir, "logs")
            os.makedirs(log_dir, exist_ok=True)
            
            log_file = os.path.join(
                log_dir, 
                f"tracker_{datetime.now().strftime('%Y%m%d')}.log"
            )
            
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_format = logging.Formatter(
                '%(asctime)s [%(levelname)s] - <%(name)s:%(lineno)d> - %(message)s'
            )
            file_handler.setFormatter(file_format)
            self.logger.addHandler(file_handler)
    
    # ---------- Simple Wrappers ----------
    def debug(self, message: str) -> None: self.logger.debug(message)
    def info(self, message: str) -> None: self.logger.info(message)
    def warning(self, message: str) -> None: self.logger.warning(message)
    def error(self, message: str) -> None: self.logger.error(message)
    def critical(self, message: str) -> None: self.logger.critical(message)

    # ---------- Domain-Specific Logs ----------
    def log_person_match(self, person_name: str, distance: float) -> None:
        if self.config.get('log_matches', True):
            self.info(f"✅ Matched face to {person_name} (distance: {distance:.3f})")
    
    def log_detection(self, detection_type: str, count: int) -> None:
        if self.config.get('log_detections', True):
            self.info(f"🔍 Detected {count} {detection_type}")
    
    def log_tracking_event(self, event_type: str, details: Dict[str, Any]) -> None:
        if self.config.get('log_tracking_events', True):
            details_str = json.dumps(details, indent=2)
            self.info(f"🎯 {event_type}:\n{details_str}")
    
    def log_plugin_result(self, plugin_name: str, person_id: int, result: Dict[str, Any]) -> None:
        if self.config.get('log_plugin_results', True):
            result_str = json.dumps(result, indent=2)
            self.info(f"🔌 Plugin '{plugin_name}' result for person {person_id}:\n{result_str}")

# Global logger instance
logger_manager = None
global_config = None

def get_logger(config: Optional[Dict[str, Any]] = None) -> LoggerManager:
    """
    Get the global logger instance.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        LoggerManager instance
    """
    global logger_manager, global_config
    
    # If config is provided, update the global config
    if config is not None:
        global_config = config
    
    # If logger_manager doesn't exist, create it with the global config.
    # If no global config has been set yet, try to load from app config.
    if logger_manager is None:
        if global_config is None:
            try:
                # Lazy import to avoid circulars; falls back silently if unavailable
                from uv_app.config import LOGGING_CONFIG  # type: ignore
                global_config = LOGGING_CONFIG
            except Exception:
                global_config = {}
        logger_manager = LoggerManager(config=global_config)
    return logger_manager

def setup_logging(config: Optional[Dict[str, Any]] = None) -> LoggerManager:
    """
    Setup and configure logging system.
    
    Args:
        config: Configuration dictionary for logging
        
    Returns:
        LoggerManager instance
    """
    global logger_manager, global_config
    global_config = config
    logger_manager = LoggerManager(config=config)
    return logger_manager
