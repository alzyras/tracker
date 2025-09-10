# uv_app/core/data_manager.py

import os
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import cv2
from .logging import get_logger

logger = get_logger()


class PersonData:
    """Container for all data related to a tracked person."""
    
    def __init__(self, track_id: int, save_dir: str = "tracked_people"):
        self.track_id = track_id
        self.save_dir = save_dir
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        
        # Person identification data
        self.name: Optional[str] = None
        self.face_encodings: List[np.ndarray] = []
        self.face_images: List[np.ndarray] = []
        self.face_boxes: List[Tuple] = []  # (top, right, bottom, left)
        
        # Body and pose data
        self.body_boxes: List[Tuple] = []  # (x1, y1, x2, y2)
        self.pose_landmarks: List[Any] = []  # list of pose objects
        
        # Current state data
        self.current_face_image: Optional[np.ndarray] = None
        self.current_body_image: Optional[np.ndarray] = None
        self.current_pose_landmarks: Optional[Any] = None
        self.current_face_bbox: Optional[Tuple] = None
        self.current_body_bbox: Optional[Tuple] = None
        
        # Plugin results
        self.plugin_results: Dict[str, List[Dict[str, Any]]] = {}
        
        # Statistics
        self.first_seen = datetime.now()
        self.last_seen = datetime.now()
        self.frame_count = 0
        self.missed_frames = 0
    
    def update_timestamp(self) -> None:
        """Update the last seen timestamp."""
        self.last_seen = datetime.now()
        self.frame_count += 1
    
    def add_face_data(self, encoding: np.ndarray, face_img: np.ndarray, 
                     bbox: Optional[Tuple] = None) -> None:
        """Add face data to the person."""
        self.face_encodings.append(encoding)
        self.face_images.append(face_img)
        if bbox:
            self.face_boxes.append(bbox)
        self.updated_at = datetime.now()
        logger.debug(f"Added face data for person {self.track_id}")
    
    def add_body_data(self, body_bbox: Optional[Tuple] = None, 
                     pose: Optional[Any] = None, 
                     body_img: Optional[np.ndarray] = None) -> None:
        """Add body and pose data to the person."""
        if body_bbox:
            self.body_boxes.append(body_bbox)
        if pose:
            self.pose_landmarks.append(pose)
        self.updated_at = datetime.now()
        logger.debug(f"Added body/pose data for person {self.track_id}")
    
    def set_name(self, name: str) -> None:
        """Set the person's name."""
        self.name = name
        self.updated_at = datetime.now()
        logger.info(f"Set name '{name}' for person {self.track_id}")
    
    def add_plugin_result(self, plugin_name: str, result: Dict[str, Any]) -> None:
        """Add a plugin result."""
        if plugin_name not in self.plugin_results:
            self.plugin_results[plugin_name] = []
        
        result_with_timestamp = {
            "timestamp": datetime.now().isoformat(),
            "result": result
        }
        self.plugin_results[plugin_name].append(result_with_timestamp)
        self.updated_at = datetime.now()
        logger.debug(f"Added plugin result from '{plugin_name}' for person {self.track_id}")
    
    def get_all_data(self) -> Dict[str, Any]:
        """Get all person data as a dictionary."""
        return {
            "track_id": self.track_id,
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "frame_count": self.frame_count,
            "missed_frames": self.missed_frames,
            "face_count": len(self.face_images),
            "body_count": len(self.body_boxes),
            "pose_count": len(self.pose_landmarks),
            "plugin_results": self.plugin_results
        }
    
    def get_current_coordinates(self) -> Dict[str, Optional[Tuple]]:
        """Get current coordinates for face and body."""
        return {
            "face_bbox": self.current_face_bbox,
            "body_bbox": self.current_body_bbox
        }


class DataManager:
    """Manages all tracked person data with persistence capabilities."""
    
    def __init__(self, save_dir: str = "tracked_people"):
        self.persons: Dict[int, PersonData] = {}
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        logger.info("Initialized DataManager")
    
    def create_person(self, track_id: int) -> PersonData:
        """Create a new person data container."""
        if track_id in self.persons:
            logger.warning(f"Person with ID {track_id} already exists")
            return self.persons[track_id]
        
        person = PersonData(track_id, self.save_dir)
        self.persons[track_id] = person
        logger.info(f"Created new person with ID {track_id}")
        return person
    
    def get_person(self, track_id: int) -> Optional[PersonData]:
        """Get person data by track ID."""
        return self.persons.get(track_id)
    
    def get_all_persons(self) -> Dict[int, PersonData]:
        """Get all tracked persons."""
        return self.persons.copy()
    
    def update_person(self, track_id: int, **kwargs) -> None:
        """Update person data."""
        person = self.get_person(track_id)
        if not person:
            logger.warning(f"Person with ID {track_id} not found")
            return
        
        for key, value in kwargs.items():
            if hasattr(person, key):
                setattr(person, key, value)
        
        person.update_timestamp()
    
    def add_plugin_result(self, track_id: int, plugin_name: str, 
                         result: Dict[str, Any]) -> None:
        """Add a plugin result to a person."""
        person = self.get_person(track_id)
        if not person:
            logger.warning(f"Person with ID {track_id} not found")
            return
        
        person.add_plugin_result(plugin_name, result)
    
    def save_person_data(self, track_id: int) -> None:
        """Save person data to disk."""
        person = self.get_person(track_id)
        if not person:
            logger.warning(f"Person with ID {track_id} not found")
            return
        
        person_dir = os.path.join(self.save_dir, f"person_{track_id}")
        os.makedirs(person_dir, exist_ok=True)
        
        # Save metadata
        metadata = person.get_all_data()
        metadata_path = os.path.join(person_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Save face encodings
        if person.face_encodings:
            encodings_path = os.path.join(person_dir, "encodings.npy")
            np.save(encodings_path, np.array(person.face_encodings))
        
        # Save face images
        for i, face_img in enumerate(person.face_images):
            img_path = os.path.join(person_dir, f"face_{i}.jpg")
            cv2.imwrite(img_path, face_img)
        
        logger.info(f"Saved data for person {track_id} to {person_dir}")
    
    def save_all_data(self) -> None:
        """Save all person data to disk."""
        for track_id in self.persons:
            self.save_person_data(track_id)
        logger.info("Saved all person data")
    
    def load_person_data(self, track_id: int) -> Optional[PersonData]:
        """Load person data from disk."""
        person_dir = os.path.join(self.save_dir, f"person_{track_id}")
        metadata_path = os.path.join(person_dir, "metadata.json")
        
        if not os.path.exists(metadata_path):
            logger.warning(f"No data found for person {track_id}")
            return None
        
        try:
            # Load metadata
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            
            # Create person object
            person = PersonData(track_id, self.save_dir)
            
            # Restore basic attributes
            person.name = metadata.get("name")
            person.created_at = datetime.fromisoformat(metadata.get("created_at"))
            person.updated_at = datetime.fromisoformat(metadata.get("updated_at"))
            person.first_seen = datetime.fromisoformat(metadata.get("first_seen"))
            person.last_seen = datetime.fromisoformat(metadata.get("last_seen"))
            person.frame_count = metadata.get("frame_count", 0)
            person.missed_frames = metadata.get("missed_frames", 0)
            
            # Load encodings if they exist
            encodings_path = os.path.join(person_dir, "encodings.npy")
            if os.path.exists(encodings_path):
                encodings = np.load(encodings_path)
                person.face_encodings = encodings.tolist()
            
            # Load plugin results
            person.plugin_results = metadata.get("plugin_results", {})
            
            # Store in manager
            self.persons[track_id] = person
            
            logger.info(f"Loaded data for person {track_id}")
            return person
            
        except Exception as e:
            logger.error(f"Error loading person {track_id}: {e}")
            return None
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all tracked persons."""
        summary = {
            "total_persons": len(self.persons),
            "persons": {}
        }
        
        for track_id, person in self.persons.items():
            summary["persons"][track_id] = {
                "name": person.name,
                "first_seen": person.first_seen.isoformat(),
                "last_seen": person.last_seen.isoformat(),
                "frame_count": person.frame_count,
                "face_count": len(person.face_images),
                "body_count": len(person.body_boxes),
                "plugin_results_count": {
                    plugin: len(results) 
                    for plugin, results in person.plugin_results.items()
                }
            }
        
        return summary
    
    def cleanup_old_data(self, max_age_days: int = 30) -> None:
        """Remove data for persons not seen in max_age_days."""
        cutoff_date = datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        cutoff_date = cutoff_date.replace(
            day=cutoff_date.day - max_age_days
        )
        
        removed_count = 0
        for track_id, person in list(self.persons.items()):
            if person.last_seen < cutoff_date:
                del self.persons[track_id]
                removed_count += 1
        
        if removed_count > 0:
            logger.info(f"Removed {removed_count} old persons from memory")