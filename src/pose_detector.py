"""
Pose Detector - Simple MediaPipe pose detection with IR preprocessing.

This module handles:
1. Frame preprocessing (CLAHE for IR cameras)
2. MediaPipe pose detection
3. Skeleton visualization

All smoothing, dropout handling, and movement calculation is delegated to
DetectionAccumulator.
"""

from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class DetectionResult:
    """Raw detection result from a single frame."""
    detected: bool
    landmarks: Optional[np.ndarray]  # Shape (33, 4): [x, y, z, visibility] per landmark
    raw_confidence: float  # MediaPipe's average visibility score
    visible_landmarks: int = 0  # How many landmarks have visibility > 0.5


class PoseDetector:
    """
    Detects body pose using MediaPipe with IR camera preprocessing.

    This is a simple detector - it processes one frame at a time and returns
    raw results. The DetectionAccumulator handles smoothing and dropout tolerance.
    """

    # MediaPipe landmark indices
    NOSE = 0
    LEFT_EYE = 2
    RIGHT_EYE = 5
    LEFT_EAR = 7
    RIGHT_EAR = 8
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28

    # Skeleton connections for visualization
    SKELETON_CONNECTIONS = [
        # Face
        (NOSE, LEFT_EYE), (NOSE, RIGHT_EYE),
        (LEFT_EYE, LEFT_EAR), (RIGHT_EYE, RIGHT_EAR),
        # Torso
        (LEFT_SHOULDER, RIGHT_SHOULDER),
        (LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP),
        (LEFT_HIP, RIGHT_HIP),
        # Arms
        (LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST),
        (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST),
        # Legs
        (LEFT_HIP, LEFT_KNEE), (LEFT_KNEE, LEFT_ANKLE),
        (RIGHT_HIP, RIGHT_KNEE), (RIGHT_KNEE, RIGHT_ANKLE),
    ]

    def __init__(
        self,
        min_detection_confidence: float = 0.3,
        min_tracking_confidence: float = 0.3,
        model_complexity: int = 1,
        enable_ir_preprocessing: bool = True,
    ):
        """
        Initialize pose detector.

        Args:
            min_detection_confidence: Minimum confidence for detection (lower = more detections but more noise)
            min_tracking_confidence: Minimum confidence for tracking
            model_complexity: 0=lite, 1=full, 2=heavy (higher = more accurate but slower)
            enable_ir_preprocessing: Whether to apply CLAHE for IR cameras
        """
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=True,  # Full detection each frame - slower but more stable for IR
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._enable_ir_preprocessing = enable_ir_preprocessing

        # CLAHE for IR preprocessing
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """
        Detect pose in a single frame.

        Args:
            frame: BGR image from camera

        Returns:
            DetectionResult with landmarks if detected
        """
        # Preprocess for IR cameras
        if self._enable_ir_preprocessing:
            frame = self._preprocess_ir(frame)

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run detection
        results = self._pose.process(rgb_frame)

        if not results.pose_landmarks:
            return DetectionResult(
                detected=False,
                landmarks=None,
                raw_confidence=0.0,
            )

        # Check how many landmarks have good visibility
        visible_count = sum(1 for lm in results.pose_landmarks.landmark if lm.visibility > 0.5)
        total_landmarks = len(results.pose_landmarks.landmark)

        # Extract landmarks as numpy array
        h, w = frame.shape[:2]
        landmarks = np.array([
            [lm.x * w, lm.y * h, lm.z * w, lm.visibility]
            for lm in results.pose_landmarks.landmark
        ])

        # Calculate average confidence
        avg_confidence = np.mean([lm.visibility for lm in results.pose_landmarks.landmark])

        return DetectionResult(
            detected=True,
            landmarks=landmarks,
            raw_confidence=avg_confidence,
            visible_landmarks=visible_count,
        )

    def _preprocess_ir(self, frame: np.ndarray) -> np.ndarray:
        """Apply preprocessing to improve detection on IR/grayscale cameras."""
        # Handle grayscale input
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # This improves local contrast which helps MediaPipe find features
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self._clahe.apply(l)
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return enhanced

    def draw_skeleton(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,
        min_confidence: float = 0.3,
        color: tuple = (0, 255, 0),
    ) -> np.ndarray:
        """
        Draw skeleton overlay on frame.

        Args:
            frame: BGR image to draw on
            landmarks: Numpy array of landmarks (33, 4)
            min_confidence: Minimum visibility to draw a landmark
            color: BGR color for skeleton

        Returns:
            Frame with skeleton drawn
        """
        output = frame.copy()

        # Draw connections
        for i1, i2 in self.SKELETON_CONNECTIONS:
            if landmarks[i1, 3] > min_confidence and landmarks[i2, 3] > min_confidence:
                pt1 = (int(landmarks[i1, 0]), int(landmarks[i1, 1]))
                pt2 = (int(landmarks[i2, 0]), int(landmarks[i2, 1]))
                cv2.line(output, pt1, pt2, color, 2)

        # Draw landmarks
        for i, lm in enumerate(landmarks):
            if lm[3] > min_confidence:
                pt = (int(lm[0]), int(lm[1]))
                # Color based on confidence
                conf_color = self._confidence_color(lm[3])
                cv2.circle(output, pt, 5, conf_color, -1)
                cv2.circle(output, pt, 5, (255, 255, 255), 1)

        return output

    def _confidence_color(self, confidence: float) -> tuple:
        """Get BGR color based on confidence (red=low, green=high)."""
        if confidence < 0.5:
            t = confidence / 0.5
            return (0, int(255 * t), 255)  # Red to yellow
        else:
            t = (confidence - 0.5) / 0.5
            return (0, 255, int(255 * (1 - t)))  # Yellow to green

    def close(self):
        """Release resources."""
        self._pose.close()
