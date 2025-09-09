"""Base analyzer class for implementing custom analyzers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np

from ...logging_config import get_logger
from ...person import TrackedPerson

logger = get_logger(__name__)


class BaseAnalyzer(ABC):
    """Abstract base class for all analyzers."""
    
    def __init__(self, name: str) -> None:
        """Initialize the analyzer.
        
        Args:
            name: Name of the analyzer
        """
        self.name = name
        self.logger = get_logger(f"analyzer.{name}")
    
    @abstractmethod
    def analyze(
        self, 
        person: TrackedPerson, 
        frame: np.ndarray,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Analyze a person in a frame.
        
        Args:
            person: The tracked person to analyze
            frame: The current frame
            **kwargs: Additional arguments specific to the analyzer
            
        Returns:
            Dictionary containing analysis results
        """
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get analyzer configuration.
        
        Returns:
            Dictionary containing analyzer configuration
        """
        pass
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame before analysis (override if needed).
        
        Args:
            frame: Input frame
            
        Returns:
            Preprocessed frame
        """
        return frame
    
    def postprocess_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Postprocess analysis result (override if needed).
        
        Args:
            result: Raw analysis result
            
        Returns:
            Processed result
        """
        return result
    
    def should_analyze(self, person: TrackedPerson) -> bool:
        """Check if this person should be analyzed (override if needed).
        
        Args:
            person: The tracked person
            
        Returns:
            True if analysis should be performed
        """
        return True
    
    def log_result(self, person_id: int, result: Dict[str, Any]) -> None:
        """Log analysis result.
        
        Args:
            person_id: ID of the analyzed person
            result: Analysis result
        """
        from ...logging_config import log_analyzer_result
        log_analyzer_result(self.logger, person_id, self.name, result)


class AnalyzerManager:
    """Manages multiple analyzers and coordinates their execution."""
    
    def __init__(self) -> None:
        """Initialize the analyzer manager."""
        self.analyzers: Dict[str, BaseAnalyzer] = {}
        self.logger = get_logger(__name__)
    
    def register_analyzer(self, analyzer: BaseAnalyzer) -> None:
        """Register an analyzer.
        
        Args:
            analyzer: The analyzer to register
        """
        self.analyzers[analyzer.name] = analyzer
        self.logger.info("Registered analyzer: %s", analyzer.name)
    
    def unregister_analyzer(self, name: str) -> None:
        """Unregister an analyzer.
        
        Args:
            name: Name of the analyzer to unregister
        """
        if name in self.analyzers:
            del self.analyzers[name]
            self.logger.info("Unregistered analyzer: %s", name)
    
    def analyze_person(
        self, 
        person: TrackedPerson, 
        frame: np.ndarray,
        analyzer_names: Optional[list] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Analyze a person using registered analyzers.
        
        Args:
            person: The tracked person to analyze
            frame: The current frame
            analyzer_names: List of specific analyzers to run (None for all)
            
        Returns:
            Dictionary mapping analyzer names to their results
        """
        results = {}
        
        # Determine which analyzers to run
        analyzers_to_run = (
            analyzer_names if analyzer_names 
            else list(self.analyzers.keys())
        )
        
        for analyzer_name in analyzers_to_run:
            if analyzer_name not in self.analyzers:
                self.logger.warning("Analyzer not found: %s", analyzer_name)
                continue
                
            analyzer = self.analyzers[analyzer_name]
            
            if not analyzer.should_analyze(person):
                continue
            
            try:
                # Preprocess frame
                processed_frame = analyzer.preprocess_frame(frame)
                
                # Run analysis
                result = analyzer.analyze(person, processed_frame)
                
                # Postprocess result
                result = analyzer.postprocess_result(result)
                
                # Store result
                results[analyzer_name] = result
                
                # Log result
                analyzer.log_result(person.track_id, result)
                
            except Exception as e:
                self.logger.error(
                    "Error in analyzer %s for person %d: %s", 
                    analyzer_name, 
                    person.track_id, 
                    e,
                )
                results[analyzer_name] = {"error": str(e)}
        
        return results
    
    def get_analyzer_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get configurations of all registered analyzers.
        
        Returns:
            Dictionary mapping analyzer names to their configurations
        """
        return {
            name: analyzer.get_config() 
            for name, analyzer in self.analyzers.items()
        }