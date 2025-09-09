# uv_app/core/recognition.py

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from config import MATCH_THRESHOLD, CANDIDATE_THRESHOLD, MIN_FRAMES_TO_CONFIRM, VERBOSE_TRACKING
from .person import TrackedPerson


class FaceRecognizer:
    """Handles face recognition and matching logic."""
    
    def __init__(self):
        self.tracked_people: List[TrackedPerson] = []
        self.lost_people: List[TrackedPerson] = []
        self.candidate_faces: List[Dict[str, Any]] = []
        self.next_id = 1
    
    def load_existing_people(self) -> None:
        """Load existing tracked people from storage."""
        from .storage import PersonStorage
        
        storage = PersonStorage()
        self.tracked_people, self.next_id = storage.load_all_people()
        
        if self.tracked_people and VERBOSE_TRACKING:
            print(f"👥 Loaded {len(self.tracked_people)} previously tracked people")
            print("   (They will be marked as 'left camera view' if not currently visible)")
    
    def find_best_match(self, face_encoding: np.ndarray) -> Tuple[Optional[TrackedPerson], float]:
        """
        Find the best matching person for a face encoding.
        
        Args:
            face_encoding: Face encoding to match
            
        Returns:
            Tuple of (best_match_person, distance)
        """
        best_match = None
        best_distance = float("inf")
        
        for person in self.tracked_people + self.lost_people:
            if person.mean_encoding is not None:
                # Consider all people, but be more lenient with those who have fewer face encodings
                distance = np.linalg.norm(face_encoding - person.mean_encoding)
                
                # Apply a small bonus for people with more face data (more reliable)
                if len(person.face_encodings) >= 3:
                    distance *= 0.95  # 5% bonus for people with 3+ faces
                elif len(person.face_encodings) == 1:
                    distance *= 1.05  # 5% penalty for people with only 1 face
                
                if distance < best_distance:
                    best_distance = distance
                    best_match = person
        
        return best_match, best_distance
    
    def process_face(self, face_encoding: np.ndarray, face_img: np.ndarray, 
                    bbox: Tuple, frame: np.ndarray) -> Optional[TrackedPerson]:
        """
        Process a detected face and either match to existing person or create new one.
        
        Args:
            face_encoding: Face encoding
            face_img: Face image ROI
            bbox: Face bounding box
            frame: Full frame for display
            
        Returns:
            Matched or created person, or None
        """
        best_match, best_distance = self.find_best_match(face_encoding)
        
        if best_match and best_distance < MATCH_THRESHOLD:
            # Match found - update existing person
            if VERBOSE_TRACKING:
                person_name = best_match.name if best_match.name else f"Person {best_match.track_id}"
                print(f"✅ Matched face to {person_name} (distance: {best_distance:.3f})")
            self._update_existing_person(best_match, face_encoding, face_img, bbox)
            return best_match
        else:
            # No match - add to candidates
            if VERBOSE_TRACKING:
                print(f"🆕 New face detected (best distance: {best_distance:.3f}, threshold: {MATCH_THRESHOLD})")
            self._add_candidate(face_encoding, face_img, bbox)
            return None
    
    def _update_existing_person(self, person: TrackedPerson, face_encoding: np.ndarray,
                               face_img: np.ndarray, bbox: Tuple) -> None:
        """Update an existing person with new face data."""
        # Move from lost to tracked if needed
        if person in self.lost_people:
            self.lost_people.remove(person)
            self.tracked_people.append(person)
            if VERBOSE_TRACKING:
                person_name = person.name if person.name else f"Person {person.track_id}"
                print(f"📥 {person_name} returned to camera view")
        
        person.update()
        person.add_face_data(face_encoding, face_img, bbox)
    
    def _add_candidate(self, face_encoding: np.ndarray, face_img: np.ndarray, 
                      bbox: Tuple) -> None:
        """Add a face to candidate list for potential new person creation."""
        self.candidate_faces.append({
            "encoding": face_encoding,
            "img": face_img,
            "bbox": bbox,
            "count": 1
        })
    
    def process_candidates(self, frame: np.ndarray) -> List[TrackedPerson]:
        """
        Process candidate faces and create new people if needed.
        
        Args:
            frame: Full frame for display
            
        Returns:
            List of newly created people
        """
        new_people = []
        
        for candidate in self.candidate_faces[:]:  # Use slice to avoid modification during iteration
            candidate["count"] += 1
            
            # Check if candidate matches any existing person
            best_match, best_distance = self.find_best_match(candidate["encoding"])
            
            if best_match and best_distance < MATCH_THRESHOLD:
                # Match found - add to existing person
                if VERBOSE_TRACKING:
                    person_name = best_match.name if best_match.name else f"Person {best_match.track_id}"
                    print(f"🔄 Candidate matched to existing {person_name} (distance: {best_distance:.3f})")
                self._update_existing_person(
                    best_match, 
                    candidate["encoding"], 
                    candidate["img"], 
                    candidate["bbox"]
                )
                self.candidate_faces.remove(candidate)
            elif candidate["count"] >= MIN_FRAMES_TO_CONFIRM:
                # Create new person
                if VERBOSE_TRACKING:
                    print(f"👤 Creating new person after {candidate['count']} frames (best distance: {best_distance:.3f})")
                new_person = self._create_new_person(
                    candidate["encoding"], 
                    candidate["img"], 
                    candidate["bbox"]
                )
                new_people.append(new_person)
                self.candidate_faces.remove(candidate)
        
        # Clean up old candidates
        self.candidate_faces = [
            c for c in self.candidate_faces 
            if c["count"] < MIN_FRAMES_TO_CONFIRM * 2
        ]
        
        return new_people
    
    def _create_new_person(self, face_encoding: np.ndarray, face_img: np.ndarray, 
                          bbox: Tuple) -> TrackedPerson:
        """Create a new tracked person."""
        new_person = TrackedPerson(self.next_id)
        new_person.add_face_data(face_encoding, face_img, bbox)
        new_person.save_data()  # Save immediately to disk
        
        self.tracked_people.append(new_person)
        self.next_id += 1
        
        print(f"New person detected! ID: {new_person.track_id}")
        return new_person
    
    def update_tracking(self, current_frame_ids: set) -> None:
        """
        Update tracking state based on current frame detections.
        
        Args:
            current_frame_ids: Set of person IDs detected in current frame
        """
        # Handle lost people
        for person in self.tracked_people[:]:
            if person.track_id not in current_frame_ids:
                person.missed_frames += 1
                if person.missed_frames > 50:  # MAX_MISSED_FRAMES
                    self.tracked_people.remove(person)
                    self.lost_people.append(person)
                    if VERBOSE_TRACKING:
                        person_name = person.name if person.name else f"Person {person.track_id}"
                        print(f"📤 {person_name} left camera view (will be re-detected if they return)")
    
    def get_certainty_percentage(self, distance: float) -> float:
        """Calculate certainty percentage from distance."""
        return max(0, min(100, (1 - distance) * 100))
    
    def get_all_people(self) -> List[TrackedPerson]:
        """Get all tracked and lost people."""
        return self.tracked_people + self.lost_people
