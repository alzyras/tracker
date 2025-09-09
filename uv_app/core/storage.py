# uv_app/core/storage.py

import os
from typing import List, Tuple
from .person import TrackedPerson
from config import SAVE_DIR


class PersonStorage:
    """Handles storage and retrieval of tracked people data."""
    
    def __init__(self):
        self.save_dir = SAVE_DIR
        self._ensure_save_dir()
    
    def _ensure_save_dir(self) -> None:
        """Ensure the save directory exists."""
        os.makedirs(self.save_dir, exist_ok=True)
    
    def load_all_people(self) -> Tuple[List[TrackedPerson], int]:
        """
        Load all existing tracked people from storage.
        
        Returns:
            Tuple of (tracked_people_list, next_id)
        """
        tracked_people = []
        next_id = 1
        
        if not os.path.exists(self.save_dir):
            return tracked_people, next_id
        
        for d_name in os.listdir(self.save_dir):
            person_dir = os.path.join(self.save_dir, d_name)
            if os.path.isdir(person_dir) and d_name.startswith("person_"):
                try:
                    track_id = int(d_name.split("_")[1])
                    person = TrackedPerson(track_id)
                    if person.face_encodings:
                        tracked_people.append(person)
                    next_id = max(next_id, track_id + 1)
                except (ValueError, IndexError):
                    continue
        
        return tracked_people, next_id
    
    def save_person(self, person: TrackedPerson) -> None:
        """Save a single person's data."""
        person.save_data()
    
    def delete_person(self, track_id: int) -> bool:
        """
        Delete a person's data from storage.
        
        Args:
            track_id: ID of person to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        person_dir = os.path.join(self.save_dir, f"person_{track_id}")
        if os.path.exists(person_dir):
            import shutil
            shutil.rmtree(person_dir)
            return True
        return False
    
    def get_person_data_path(self, track_id: int) -> str:
        """Get the data directory path for a person."""
        return os.path.join(self.save_dir, f"person_{track_id}")
    
    def list_all_people(self) -> List[dict]:
        """
        List all people with their basic information.
        
        Returns:
            List of dictionaries with person information
        """
        people_info = []
        
        if not os.path.exists(self.save_dir):
            return people_info
        
        for d_name in os.listdir(self.save_dir):
            person_dir = os.path.join(self.save_dir, d_name)
            if os.path.isdir(person_dir) and d_name.startswith("person_"):
                try:
                    track_id = int(d_name.split("_")[1])
                    person = TrackedPerson(track_id)
                    
                    # Count face images
                    face_images = [f for f in os.listdir(person_dir) if f.startswith('face_') and f.endswith('.jpg')]
                    
                    people_info.append({
                        'track_id': track_id,
                        'name': person.name,
                        'num_faces': len(face_images),
                        'has_encodings': len(person.face_encodings) > 0,
                        'data_path': person_dir
                    })
                except (ValueError, IndexError):
                    continue
        
        return people_info
    
    def cleanup_empty_directories(self) -> int:
        """
        Clean up empty person directories.
        
        Returns:
            Number of directories cleaned up
        """
        cleaned = 0
        
        if not os.path.exists(self.save_dir):
            return cleaned
        
        for d_name in os.listdir(self.save_dir):
            person_dir = os.path.join(self.save_dir, d_name)
            if os.path.isdir(person_dir) and d_name.startswith("person_"):
                try:
                    # Check if directory is empty or has no face images
                    files = os.listdir(person_dir)
                    face_images = [f for f in files if f.startswith('face_') and f.endswith('.jpg')]
                    
                    if not face_images or len(files) == 0:
                        import shutil
                        shutil.rmtree(person_dir)
                        cleaned += 1
                        print(f"Cleaned up empty directory: {person_dir}")
                except Exception as e:
                    print(f"Error cleaning up {person_dir}: {e}")
        
        return cleaned
