"""Extra processing for body detection and pose estimation."""

from typing import List, Tuple

import cv2
import mediapipe as mp
import numpy as np

from .config import (
    BODY_BOX_COLOR,
    BOX_THICKNESS,
    POSE_COLOR,
    POSE_CONNECTION_COLOR,
    POSE_CONNECTION_THICKNESS,
    POSE_LANDMARK_RADIUS,
    SHOW_BODY_BOXES,
    SHOW_POSES,
)
from .logging_config import get_logger

logger = get_logger(__name__)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


class ExtrasProcessor:
    """Handles full-body bounding boxes and pose landmarks."""

    def __init__(
        self,
        enable_body: bool = True,
        enable_pose: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        """Initialize the extras processor.
        
        Args:
            enable_body: Enable body bounding box detection
            enable_pose: Enable pose landmark detection
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
        """
        self.enable_body = enable_body
        self.enable_pose = enable_pose
        self.pose_model = mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        logger.info(
            "Initialized ExtrasProcessor - body: %s, pose: %s",
            enable_body,
            enable_pose,
        )

    def process_frame(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]], List]:
        """Process frame for body and pose detection.
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (annotated_frame, body_boxes, pose_landmarks)
        """
        bodies = []
        poses = []
        annotated = frame.copy()

        if self.enable_pose or self.enable_body:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose_model.process(rgb)

            if results.pose_landmarks:
                if self.enable_pose and SHOW_POSES:
                    # Draw pose landmarks with configured colors
                    mp_drawing.draw_landmarks(
                        annotated,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(
                            color=POSE_COLOR,
                            thickness=BOX_THICKNESS,
                            circle_radius=POSE_LANDMARK_RADIUS,
                        ),
                        connection_drawing_spec=mp_drawing.DrawingSpec(
                            color=POSE_CONNECTION_COLOR,
                            thickness=POSE_CONNECTION_THICKNESS,
                        ),
                    )

                if self.enable_body:
                    h, w, _ = frame.shape
                    xs = [lm.x * w for lm in results.pose_landmarks.landmark]
                    ys = [lm.y * h for lm in results.pose_landmarks.landmark]
                    x1, y1 = int(min(xs)), int(min(ys))
                    x2, y2 = int(max(xs)), int(max(ys))
                    bodies.append((x1, y1, x2, y2))

                    # Draw body bounding box if enabled
                    if SHOW_BODY_BOXES:
                        cv2.rectangle(
                            annotated,
                            (x1, y1),
                            (x2, y2),
                            BODY_BOX_COLOR,
                            BOX_THICKNESS,
                        )

                poses.append(results.pose_landmarks)

        return annotated, bodies, poses

    def __del__(self) -> None:
        """Clean up resources."""
        if hasattr(self, "pose_model"):
            self.pose_model.close()
