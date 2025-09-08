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
                    label = f"ID {best_match.track_id}"
                    if best_match.name: label += f" ({best_match.name})"
                    draw_face_box(frame, (top,right,bottom,left), label)
                else:
                    candidate_faces.append({"encoding": face_encoding, "img": frame[top:bottom,left:right], "count":1})

        # Extras
        if enable_body or enable_pose:
            frame, bodies, poses = extras.process_frame(frame)
            for i, person in enumerate(tracked_people):
                # Assign the latest body box and pose to each tracked person (simplified)
                if bodies: person.add_body_data(bodies[-1], poses[-1] if poses else None)

        cv2.imshow("Tracker", frame)
        if cv2.waitKey(1) & 0xFF in (27, ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()