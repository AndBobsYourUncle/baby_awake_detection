"""
MediaPipe-based pose estimator implementing the PoseEstimator interface.

This serves as a fallback option and reference implementation.
"""

from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from .pose_api import (
    PoseEstimator,
    PoseResult,
    Keypoint,
    KeypointName,
)
from .config import PoseConfig


# Mapping from MediaPipe landmark indices to our KeypointName enum
MEDIAPIPE_TO_KEYPOINT = {
    0: KeypointName.NOSE,
    2: KeypointName.LEFT_EYE,
    5: KeypointName.RIGHT_EYE,
    7: KeypointName.LEFT_EAR,
    8: KeypointName.RIGHT_EAR,
    11: KeypointName.LEFT_SHOULDER,
    12: KeypointName.RIGHT_SHOULDER,
    13: KeypointName.LEFT_ELBOW,
    14: KeypointName.RIGHT_ELBOW,
    15: KeypointName.LEFT_WRIST,
    16: KeypointName.RIGHT_WRIST,
    23: KeypointName.LEFT_HIP,
    24: KeypointName.RIGHT_HIP,
    25: KeypointName.LEFT_KNEE,
    26: KeypointName.RIGHT_KNEE,
    27: KeypointName.LEFT_ANKLE,
    28: KeypointName.RIGHT_ANKLE,
}


class MediaPipeEstimator(PoseEstimator):
    """
    MediaPipe Pose-based implementation of PoseEstimator.

    Provides a working implementation while ViTPose is being set up,
    or as a lightweight fallback for systems without GPU.
    """

    def __init__(self, config: PoseConfig):
        """
        Initialize MediaPipe pose estimator.

        Args:
            config: Pose configuration
        """
        self._config = config

        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=config.mediapipe_model_complexity,
            min_detection_confidence=config.mediapipe_min_detection_confidence,
            min_tracking_confidence=config.mediapipe_min_tracking_confidence,
        )

        # CLAHE for IR preprocessing
        if config.enable_ir_preprocessing:
            self._clahe = cv2.createCLAHE(
                clipLimit=config.clahe_clip_limit,
                tileGridSize=(config.clahe_grid_size, config.clahe_grid_size),
            )
        else:
            self._clahe = None

    def estimate(self, frame_bgr: np.ndarray) -> Optional[PoseResult]:
        """
        Estimate pose from a BGR frame.

        Args:
            frame_bgr: OpenCV BGR image

        Returns:
            PoseResult if pose detected, None otherwise
        """
        # Preprocess for IR cameras
        if self._clahe is not None:
            frame_bgr = self._preprocess_ir(frame_bgr)

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Run detection
        results = self._pose.process(rgb_frame)

        if not results.pose_landmarks:
            return None

        # Convert to our format
        h, w = frame_bgr.shape[:2]
        keypoints = {}

        for mp_idx, kp_name in MEDIAPIPE_TO_KEYPOINT.items():
            lm = results.pose_landmarks.landmark[mp_idx]
            keypoints[kp_name] = Keypoint(
                name=kp_name,
                x=lm.x * w,
                y=lm.y * h,
                confidence=lm.visibility,
            )

        # Calculate overall score as average visibility
        avg_confidence = np.mean([lm.visibility for lm in results.pose_landmarks.landmark])

        return PoseResult(
            keypoints=keypoints,
            score=avg_confidence,
        )

    def _preprocess_ir(self, frame: np.ndarray) -> np.ndarray:
        """Apply preprocessing to improve detection on IR/grayscale cameras."""
        # Handle grayscale input
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self._clahe.apply(l)
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return enhanced

    def close(self):
        """Release resources."""
        self._pose.close()
