# uv_app/actions.py

import cv2
import numpy as np
from deepface import DeepFace
import mediapipe as mp
import math

mp_pose = mp.solutions.pose

class ActionsProcessor:
    """
    Detect emotions and body activities for each person.
    """

    def __init__(self, enable_emotions=True, enable_phone=True, enable_actions=True):
        self.enable_emotions = enable_emotions
        self.enable_phone = enable_phone
        self.enable_actions = enable_actions

    def analyze_face(self, face_img):
        """Return dominant emotion."""
        if not self.enable_emotions:
            return None
        try:
            result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
            return result['dominant_emotion']
        except:
            return None

    def analyze_phone(self, pose_landmarks):
        """Detect if person is likely looking at phone (head down + hands near face)."""
        if not self.enable_phone or not pose_landmarks:
            return False
        nose = pose_landmarks.landmark[0]
        left_shoulder = pose_landmarks.landmark[11]
        right_shoulder = pose_landmarks.landmark[12]
        angle = abs(nose.y - (left_shoulder.y + right_shoulder.y)/2)
        return angle > 0.15

    def analyze_body_actions(self, pose_landmarks):
        """
        Detect common body actions using pose landmarks.
        Returns a dict: { 'waving':bool, 'pointing':bool, 'texting':bool, ... }
        """
        if not self.enable_actions or not pose_landmarks:
            return {}

        actions = {}
        # Example: waving = hand above shoulder + moving horizontally (simplified)
        left_wrist = pose_landmarks.landmark[15]
        right_wrist = pose_landmarks.landmark[16]
        left_shoulder = pose_landmarks.landmark[11]
        right_shoulder = pose_landmarks.landmark[12]

        actions['waving'] = left_wrist.y < left_shoulder.y or right_wrist.y < right_shoulder.y
        # Example: texting/phone: hand near face + head down
        nose = pose_landmarks.landmark[0]
        actions['texting'] = ((abs(left_wrist.x - nose.x) < 0.1 and abs(left_wrist.y - nose.y) < 0.1) or
                              (abs(right_wrist.x - nose.x) < 0.1 and abs(right_wrist.y - nose.y) < 0.1))
        # Additional actions can be added here
        return actions