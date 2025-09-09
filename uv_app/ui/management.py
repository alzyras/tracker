# uv_app/ui/management.py

import os
import json
import cv2
from typing import List, Dict, Optional, Tuple
from core.storage import PersonStorage
from core.person import TrackedPerson


class PeopleManager:
    """Manages tracked people - viewing, renaming, organizing."""
    
    def __init__(self):
        self.storage = PersonStorage()
    
    def list_people(self) -> List[Dict]:
        """
        List all tracked people with their information.
        
        Returns:
            List of dictionaries with person information
        """
        return self.storage.list_all_people()
    
    def get_person_details(self, track_id: int) -> Optional[Dict]:
        """
        Get detailed information about a specific person.
        
        Args:
            track_id: Person's track ID
            
        Returns:
            Dictionary with person details or None if not found
        """
        person = TrackedPerson(track_id)
        if not person.face_encodings:
            return None
        
        person_dir = self.storage.get_person_data_path(track_id)
        face_images = [f for f in os.listdir(person_dir) if f.startswith('face_') and f.endswith('.jpg')]
        
        return {
            'track_id': track_id,
            'name': person.name,
            'num_face_encodings': len(person.face_encodings),
            'num_face_images': len(face_images),
            'emotion_history': person.emotion_history,
            'phone_history': person.phone_history,
            'body_actions': person.body_actions,
            'face_images': face_images,
            'data_path': person_dir
        }
    
    def rename_person(self, track_id: int, new_name: str) -> bool:
        """
        Rename a person.
        
        Args:
            track_id: Person's track ID
            new_name: New name for the person
            
        Returns:
            True if successful, False otherwise
        """
        try:
            person = TrackedPerson(track_id)
            if person.face_encodings:  # Only rename if person exists
                person.set_name(new_name)
                return True
        except Exception as e:
            print(f"Error renaming person {track_id}: {e}")
        return False
    
    def delete_person(self, track_id: int) -> bool:
        """
        Delete a person and all their data.
        
        Args:
            track_id: Person's track ID
            
        Returns:
            True if successful, False otherwise
        """
        return self.storage.delete_person(track_id)
    
    def view_face_images(self, track_id: int) -> List[str]:
        """
        Get list of face image files for a person.
        
        Args:
            track_id: Person's track ID
            
        Returns:
            List of face image filenames
        """
        person_dir = self.storage.get_person_data_path(track_id)
        if not os.path.exists(person_dir):
            return []
        
        face_images = [f for f in os.listdir(person_dir) if f.startswith('face_') and f.endswith('.jpg')]
        return sorted(face_images)
    
    def display_face_image(self, track_id: int, image_filename: str) -> None:
        """
        Display a face image in a window.
        
        Args:
            track_id: Person's track ID
            image_filename: Name of the face image file
        """
        person_dir = self.storage.get_person_data_path(track_id)
        image_path = os.path.join(person_dir, image_filename)
        
        if os.path.exists(image_path):
            img = cv2.imread(image_path)
            if img is not None:
                cv2.imshow(f"Person {track_id} - {image_filename}", img)
                print(f"Press any key to close the image window...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print(f"Could not load image: {image_path}")
        else:
            print(f"Image not found: {image_path}")
    
    def cleanup_empty_people(self) -> int:
        """
        Clean up people with no face data.
        
        Returns:
            Number of people cleaned up
        """
        return self.storage.cleanup_empty_directories()
    
    def export_person_data(self, track_id: int, export_path: str) -> bool:
        """
        Export person data to a JSON file.
        
        Args:
            track_id: Person's track ID
            export_path: Path to save the export file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            person = TrackedPerson(track_id)
            if not person.face_encodings:
                return False
            
            export_data = {
                'track_id': track_id,
                'name': person.name,
                'num_faces': len(person.face_images),
                'emotion_history': person.emotion_history,
                'phone_history': person.phone_history,
                'body_actions': person.body_actions,
                'face_images': self.view_face_images(track_id)
            }
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error exporting person data: {e}")
            return False


class DuplicateCleaner:
    """Handles finding and merging duplicate people."""
    
    def __init__(self):
        self.storage = PersonStorage()
    
    def find_duplicates(self, similarity_threshold: float = 0.3) -> List[List[int]]:
        """
        Find potential duplicate people based on face similarity.
        
        Args:
            similarity_threshold: Threshold for considering faces similar
            
        Returns:
            List of lists, each containing track IDs of potential duplicates
        """
        import numpy as np
        
        people_info = self.storage.list_all_people()
        duplicates = []
        processed = set()
        
        for i, person1_info in enumerate(people_info):
            if person1_info['track_id'] in processed:
                continue
            
            person1 = TrackedPerson(person1_info['track_id'])
            if not person1.face_encodings:
                continue
            
            duplicate_group = [person1_info['track_id']]
            
            for j, person2_info in enumerate(people_info[i+1:], i+1):
                if person2_info['track_id'] in processed:
                    continue
                
                person2 = TrackedPerson(person2_info['track_id'])
                if not person2.face_encodings:
                    continue
                
                # Calculate similarity between mean encodings
                if person1.mean_encoding is not None and person2.mean_encoding is not None:
                    distance = np.linalg.norm(person1.mean_encoding - person2.mean_encoding)
                    if distance < similarity_threshold:
                        duplicate_group.append(person2_info['track_id'])
                        processed.add(person2_info['track_id'])
            
            if len(duplicate_group) > 1:
                duplicates.append(duplicate_group)
                processed.update(duplicate_group)
        
        return duplicates
    
    def merge_duplicates(self, duplicate_group: List[int], keep_id: int) -> bool:
        """
        Merge duplicate people into one.
        
        Args:
            duplicate_group: List of track IDs to merge
            keep_id: ID to keep (merge others into this one)
            
        Returns:
            True if successful, False otherwise
        """
        if keep_id not in duplicate_group:
            return False
        
        try:
            # Load the person to keep
            main_person = TrackedPerson(keep_id)
            
            # Merge data from other people
            for track_id in duplicate_group:
                if track_id == keep_id:
                    continue
                
                other_person = TrackedPerson(track_id)
                if other_person.face_encodings:
                    # Add face data from other person
                    for encoding, face_img in zip(other_person.face_encodings, other_person.face_images):
                        main_person.add_face_data(encoding, face_img)
                    
                    # Merge other data
                    main_person.emotion_history.extend(other_person.emotion_history)
                    main_person.phone_history.extend(other_person.phone_history)
                    main_person.body_actions.update(other_person.body_actions)
                
                # Delete the other person
                self.storage.delete_person(track_id)
            
            # Save the merged person
            main_person.save_data()
            return True
            
        except Exception as e:
            print(f"Error merging duplicates: {e}")
            return False
