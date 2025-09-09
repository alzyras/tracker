from extras import ExtrasProcessor
from person import TrackedPerson
from utils import resize_frame, draw_face_box
import cv2
import face_recognition
import numpy as np
import os
from config import SAVE_DIR, MAX_MISSED_FRAMES, MIN_FRAMES_TO_CONFIRM, MATCH_THRESHOLD, CANDIDATE_THRESHOLD

def load_existing_people():
    tracked_people = []
    next_id = 1
    for d_name in os.listdir(SAVE_DIR):
        person_dir = os.path.join(SAVE_DIR, d_name)
        if os.path.isdir(person_dir):
            try:
                track_id = int(d_name.split("_")[1])
                person = TrackedPerson(track_id)
                if person.face_encodings:
                    tracked_people.append(person)
                next_id = max(next_id, track_id + 1)
            except: continue
    return tracked_people, next_id

def run_tracker(video_source=0, enable_face=True, enable_body=True, enable_pose=True):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source {video_source}")

    tracked_people, next_id = load_existing_people()
    lost_people = []
    candidate_faces = []

    extras = ExtrasProcessor(enable_body=enable_body, enable_pose=enable_pose)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = resize_frame(frame)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        current_frame_ids = set()

        if enable_face:
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            for i, (top,right,bottom,left) in enumerate(face_locations):
                face_encoding = face_encodings[i]
                best_match, best_distance = None, float("inf")
                for person in tracked_people + lost_people:
                    if person.mean_encoding is not None:
                        d = np.linalg.norm(face_encoding - person.mean_encoding)
                        if d < best_distance:
                            best_distance = d
                            best_match = person

                if best_match and best_distance < MATCH_THRESHOLD:
                    if best_match in lost_people:
                        lost_people.remove(best_match)
                        tracked_people.append(best_match)
                    best_match.update()
                    best_match.add_face_data(face_encoding, frame[top:bottom,left:right], bbox=(top,right,bottom,left))
                    current_frame_ids.add(best_match.track_id)
                    
                    # Calculate certainty percentage (lower distance = higher certainty)
                    certainty = max(0, min(100, (1 - best_distance) * 100))
                    label = f"ID {best_match.track_id} ({certainty:.1f}%)"
                    if best_match.name: 
                        label += f" - {best_match.name}"
                    
                    draw_face_box(frame, (top,right,bottom,left), label)
                else:
                    # Add to candidate faces
                    candidate_faces.append({
                        "encoding": face_encoding, 
                        "img": frame[top:bottom,left:right], 
                        "bbox": (top,right,bottom,left),
                        "count": 1
                    })

        # Process candidate faces - check against existing people first
        for candidate in candidate_faces[:]:  # Use slice to avoid modification during iteration
            candidate["count"] += 1
            
            # Check if this candidate matches any existing person (including lost ones)
            best_match, best_distance = None, float("inf")
            for person in tracked_people + lost_people:
                if person.mean_encoding is not None:
                    d = np.linalg.norm(candidate["encoding"] - person.mean_encoding)
                    if d < best_distance:
                        best_distance = d
                        best_match = person
            
            # If candidate matches an existing person, add to that person
            if best_match and best_distance < CANDIDATE_THRESHOLD:
                if best_match in lost_people:
                    lost_people.remove(best_match)
                    tracked_people.append(best_match)
                    print(f"Re-found person {best_match.track_id} (was lost)")
                best_match.update()
                best_match.add_face_data(candidate["encoding"], candidate["img"], candidate["bbox"])
                current_frame_ids.add(best_match.track_id)
                
                # Calculate certainty percentage
                certainty = max(0, min(100, (1 - best_distance) * 100))
                label = f"ID {best_match.track_id} ({certainty:.1f}%)"
                if best_match.name: 
                    label += f" - {best_match.name}"
                
                draw_face_box(frame, candidate["bbox"], label)
                candidate_faces.remove(candidate)
            
            # If candidate has been seen enough times and doesn't match anyone, create new person
            elif candidate["count"] >= MIN_FRAMES_TO_CONFIRM:
                new_person = TrackedPerson(next_id)
                new_person.add_face_data(candidate["encoding"], candidate["img"], candidate["bbox"])
                new_person.save_data()  # Save immediately to disk
                tracked_people.append(new_person)
                current_frame_ids.add(next_id)
                
                print(f"New person detected! ID: {next_id}")
                label = f"NEW ID {next_id}"
                draw_face_box(frame, candidate["bbox"], label, color=(0, 0, 255))  # Red for new person
                
                next_id += 1
                candidate_faces.remove(candidate)

        # Extras
        if enable_body or enable_pose:
            frame, bodies, poses = extras.process_frame(frame)
            for i, person in enumerate(tracked_people):
                # Assign the latest body box and pose to each tracked person (simplified)
                if bodies: person.add_body_data(bodies[-1], poses[-1] if poses else None)

        # Handle lost people
        for person in tracked_people[:]:
            if person.track_id not in current_frame_ids:
                person.missed_frames += 1
                if person.missed_frames > MAX_MISSED_FRAMES:
                    tracked_people.remove(person)
                    lost_people.append(person)
                    print(f"Person {person.track_id} lost (moved to lost list)")

        # Clean up old candidates
        candidate_faces = [c for c in candidate_faces if c["count"] < MIN_FRAMES_TO_CONFIRM * 2]

        cv2.imshow("Tracker", frame)
        if cv2.waitKey(1) & 0xFF in (27, ord('q')):
            break

    # Save all tracked people data before exiting
    for person in tracked_people + lost_people:
        person.save_data()
    
    print(f"Saved data for {len(tracked_people)} tracked people and {len(lost_people)} lost people")
    
    cap.release()
    cv2.destroyAllWindows()