"""Main tracking module with face recognition and person tracking."""

import os
from typing import List, Tuple

import cv2
import face_recognition
import numpy as np

from .config import (
    CANDIDATE_THRESHOLD,
    CAPTURE_INTERVALS,
    ENABLE_ANALYZERS,
    MATCH_THRESHOLD,
    MAX_MISSED_FRAMES,
    MIN_FRAMES_TO_CONFIRM,
    SAVE_DIR,
)
from .extras import ExtrasProcessor
from .logging_config import get_logger, setup_logging
from .modules.analyzers.base import AnalyzerManager
from .modules.analyzers.emotion_analyzer import EmotionAnalyzer
from .modules.analyzers.pose_analyzer import PoseAnalyzer
from .modules.capture_manager import CaptureManager
from .person import TrackedPerson
from .utils import create_person_label, draw_face_box, resize_frame, scale_frame

logger = get_logger(__name__)

def load_existing_people() -> Tuple[List[TrackedPerson], int]:
    """Load existing tracked people from disk.
    
    Returns:
        Tuple of (tracked_people, next_id)
    """
    tracked_people = []
    next_id = 1
    
    if not os.path.exists(SAVE_DIR):
        logger.info("Save directory does not exist, starting fresh")
        return tracked_people, next_id
        
    for d_name in os.listdir(SAVE_DIR):
        person_dir = os.path.join(SAVE_DIR, d_name)
        if os.path.isdir(person_dir):
            try:
                track_id = int(d_name.split("_")[1])
                person = TrackedPerson(track_id)
                if person.face_encodings:
                    tracked_people.append(person)
                    logger.info("Loaded person %d with %d face encodings", 
                              track_id, len(person.face_encodings))
                next_id = max(next_id, track_id + 1)
            except (ValueError, IndexError) as e:
                logger.warning("Failed to load person from %s: %s", d_name, e)
                continue
    
    logger.info("Loaded %d existing people, next ID: %d", len(tracked_people), next_id)
    return tracked_people, next_id

def run_tracker(
    video_source=0,
    enable_face: bool = True,
    enable_body: bool = True,
    enable_pose: bool = True,
) -> None:
    """Run the main tracking loop.
    
    Args:
        video_source: Video source (camera index or file path)
        enable_face: Enable face detection and tracking
        enable_body: Enable body bounding box detection
        enable_pose: Enable pose landmark detection
    """
    # Setup logging
    setup_logging()
    logger.info("Starting tracker with video source: %s", video_source)
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source {video_source}")

    # Load existing people and initialize tracking
    tracked_people, next_id = load_existing_people()
    lost_people = []
    candidate_faces = []

    # Initialize processors
    extras = ExtrasProcessor(enable_body=enable_body, enable_pose=enable_pose)
    capture_manager = CaptureManager(CAPTURE_INTERVALS)
    
    # Initialize analyzers
    analyzer_manager = None
    if ENABLE_ANALYZERS:
        analyzer_manager = AnalyzerManager()
        try:
            analyzer_manager.register_analyzer(EmotionAnalyzer())
            analyzer_manager.register_analyzer(PoseAnalyzer())
        except Exception as e:
            logger.warning("Failed to initialize some analyzers: %s", e)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video stream")
                break
                
            # Process frame
            frame = resize_frame(frame)
            frame = scale_frame(frame)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            current_frame_ids = set()

            # Face detection and tracking
            if enable_face:
                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                
                for i, (top, right, bottom, left) in enumerate(face_locations):
                    face_encoding = face_encodings[i]
                    best_match, best_distance = None, float("inf")
                    
                    # Find best match among existing people
                    for person in tracked_people + lost_people:
                        if person.mean_encoding is not None:
                            d = np.linalg.norm(face_encoding - person.mean_encoding)
                            if d < best_distance:
                                best_distance = d
                                best_match = person

                    if best_match and best_distance < MATCH_THRESHOLD:
                        # Matched existing person
                        if best_match in lost_people:
                            lost_people.remove(best_match)
                            tracked_people.append(best_match)
                            logger.info("Recovered lost person %d", best_match.track_id)
                        
                        best_match.update()
                        face_img = frame[top:bottom, left:right]
                        best_match.add_face_data(
                            face_encoding, face_img, bbox=(top, right, bottom, left)
                        )
                        current_frame_ids.add(best_match.track_id)
                        
                        # Create label and draw face box
                        label = create_person_label(best_match.track_id, best_match.name)
                        draw_face_box(frame, (top, right, bottom, left), label)
                        
                        # Run analyzers
                        if analyzer_manager:
                            analyzer_manager.analyze_person(best_match, frame)
                            
                        # Capture snapshots
                        capture_manager.maybe_capture_head(best_match, frame)
                        
                    else:
                        # New candidate face
                        candidate_faces.append({
                            "encoding": face_encoding,
                            "img": frame[top:bottom, left:right],
                            "count": 1,
                        })

            # Process candidates for new people (simplified for now)
            # You can implement more sophisticated candidate confirmation here

            # Body and pose processing
            if enable_body or enable_pose:
                frame, bodies, poses = extras.process_frame(frame)
                
                # Assign body/pose data to tracked people (simplified assignment)
                for i, person in enumerate(tracked_people):
                    if i < len(bodies):
                        person.add_body_data(bodies[i], poses[i] if i < len(poses) else None)
                        capture_manager.maybe_capture_body(person, frame)
                        capture_manager.maybe_capture_pose(person, frame)

            # Handle lost people
            for person in tracked_people[:]:  # Copy to avoid modification during iteration
                if person.track_id not in current_frame_ids:
                    person.missed_frames += 1
                    if person.missed_frames > MAX_MISSED_FRAMES:
                        tracked_people.remove(person)
                        lost_people.append(person)
                        logger.info("Lost person %d", person.track_id)

            # Display frame
            cv2.imshow("Tracker", frame)
            
            # Check for exit
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):  # ESC or 'q'
                logger.info("Exit requested by user")
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error("Error in tracking loop: %s", e)
        raise
    finally:
        # Cleanup
        logger.info("Cleaning up...")
        
        # Save all tracked people
        for person in tracked_people + lost_people:
            person.save_data()
        
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Tracker stopped")