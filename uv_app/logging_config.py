"""Logging configuration for the tracking application."""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional

from .config import (
    LOG_BACKUP_COUNT,
    LOG_DATE_FORMAT,
    LOG_FILE_PATH,
    LOG_FORMAT,
    LOG_LEVEL,
    LOG_TO_FILE,
    MAX_LOG_FILE_SIZE,
)


def setup_logging(
    log_level: Optional[str] = None,
    log_to_file: Optional[bool] = None,
    log_file_path: Optional[str] = None,
) -> logging.Logger:
    """Set up logging configuration for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to file
        log_file_path: Path to log file
        
    Returns:
        Configured logger instance
    """
    # Use config defaults if not provided
    level = log_level or LOG_LEVEL
    to_file = log_to_file if log_to_file is not None else LOG_TO_FILE
    file_path = log_file_path or LOG_FILE_PATH
    
    # Create logger
    logger = logging.getLogger("tracker")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if enabled)
    if to_file:
        # Ensure log directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Use rotating file handler to manage log file size
        file_handler = logging.handlers.RotatingFileHandler(
            file_path,
            maxBytes=MAX_LOG_FILE_SIZE,
            backupCount=LOG_BACKUP_COUNT,
        )
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific module.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f"tracker.{name}")


def log_analyzer_result(
    logger: logging.Logger,
    person_id: int,
    analyzer_name: str,
    result: dict,
) -> None:
    """Log analyzer results in a structured format.
    
    Args:
        logger: Logger instance
        person_id: ID of the tracked person
        analyzer_name: Name of the analyzer
        result: Analysis result dictionary
    """
    logger.info(
        "Analyzer result - Person: %d, Analyzer: %s, Result: %s",
        person_id,
        analyzer_name,
        result,
    )