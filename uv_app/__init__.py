import logging
from .core.logging import setup_logging
from .config import LOGGING_CONFIG, SAVE_DIR

# Setup logging with configuration
logger_manager = setup_logging(LOGGING_CONFIG)

# Setup file logging with save directory
logger_manager.setup_file_logging(SAVE_DIR)
LOGGER = logger_manager.logger
