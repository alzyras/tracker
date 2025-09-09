"""UV App - Person tracking application with face recognition and pose analysis."""

__version__ = "0.2.0"

from .logging_config import setup_logging
from .tracker import run_tracker

__all__ = ["setup_logging", "run_tracker"]
