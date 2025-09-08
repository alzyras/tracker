import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class ExtrasProcessor:
    """Handles full-body bounding boxes and pose landmarks."""

    def __init__(self, enable_body=True, enable_pose=True):
        self.enable_body = enable_body
        self.enable_pose = enable_pose
        self.pose_model = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

    def process_frame(self, frame):
        bodies = []
        poses = []
        annotated = frame.copy()

        if self.enable_pose or self.enable_body:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose_model.process(rgb)

            if results.pose_landmarks:
                if self.enable_pose:
                    mp_drawing.draw_landmarks(
                        annotated,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
                    )

                if self.enable_body:
                    h, w, _ = frame.shape
                    xs = [lm.x * w for lm in results.pose_landmarks.landmark]
                    ys = [lm.y * h for lm in results.pose_landmarks.landmark]
                    x1, y1 = int(min(xs)), int(min(ys))
                    x2, y2 = int(max(xs)), int(max(ys))
                    bodies.append((x1, y1, x2, y2))

                    # Optional: draw bounding box
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)

                poses.append(results.pose_landmarks)

        return annotated, bodies, poses
