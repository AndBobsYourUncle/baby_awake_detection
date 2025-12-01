"""
Pose API - Abstract interface for pose estimation.

This module defines the contract that all pose estimators must implement.
The rest of the pipeline only interacts with this interface, making it easy
to swap between MediaPipe, ViTPose, or fine-tuned models.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class KeypointName(str, Enum):
    """Standard keypoint names used across all pose estimators."""
    # Head
    NOSE = "nose"
    LEFT_EYE = "left_eye"
    RIGHT_EYE = "right_eye"
    LEFT_EAR = "left_ear"
    RIGHT_EAR = "right_ear"

    # Torso
    LEFT_SHOULDER = "left_shoulder"
    RIGHT_SHOULDER = "right_shoulder"
    LEFT_HIP = "left_hip"
    RIGHT_HIP = "right_hip"

    # Arms
    LEFT_ELBOW = "left_elbow"
    RIGHT_ELBOW = "right_elbow"
    LEFT_WRIST = "left_wrist"
    RIGHT_WRIST = "right_wrist"

    # Legs (less reliable for crib view but included for completeness)
    LEFT_KNEE = "left_knee"
    RIGHT_KNEE = "right_knee"
    LEFT_ANKLE = "left_ankle"
    RIGHT_ANKLE = "right_ankle"


# Keypoint groups for different analysis purposes
TORSO_KEYPOINTS = [
    KeypointName.NOSE,
    KeypointName.LEFT_SHOULDER,
    KeypointName.RIGHT_SHOULDER,
    KeypointName.LEFT_HIP,
    KeypointName.RIGHT_HIP,
]

ARM_KEYPOINTS = [
    KeypointName.LEFT_ELBOW,
    KeypointName.RIGHT_ELBOW,
    KeypointName.LEFT_WRIST,
    KeypointName.RIGHT_WRIST,
]

HEAD_KEYPOINTS = [
    KeypointName.NOSE,
    KeypointName.LEFT_EYE,
    KeypointName.RIGHT_EYE,
    KeypointName.LEFT_EAR,
    KeypointName.RIGHT_EAR,
]


@dataclass
class Keypoint:
    """A single detected keypoint."""
    name: KeypointName
    x: float  # Pixel coordinates
    y: float  # Pixel coordinates
    confidence: float  # 0.0 to 1.0


@dataclass
class PoseResult:
    """Result from pose estimation on a single frame."""
    keypoints: Dict[KeypointName, Keypoint] = field(default_factory=dict)
    score: float = 0.0  # Overall pose confidence

    def get(self, name: KeypointName) -> Optional[Keypoint]:
        """Get a keypoint by name, or None if not detected."""
        return self.keypoints.get(name)

    def get_visible(self, names: List[KeypointName], min_confidence: float = 0.3) -> List[Keypoint]:
        """Get all keypoints from a list that have sufficient confidence."""
        result = []
        for name in names:
            kp = self.keypoints.get(name)
            if kp and kp.confidence >= min_confidence:
                result.append(kp)
        return result

    def count_visible(self, names: List[KeypointName], min_confidence: float = 0.3) -> int:
        """Count how many keypoints from a list are visible."""
        return len(self.get_visible(names, min_confidence))

    @property
    def torso_keypoints(self) -> List[Keypoint]:
        """Get visible torso keypoints."""
        return self.get_visible(TORSO_KEYPOINTS)

    @property
    def arm_keypoints(self) -> List[Keypoint]:
        """Get visible arm keypoints."""
        return self.get_visible(ARM_KEYPOINTS)

    @property
    def torso_quality(self) -> float:
        """Quality score based on torso keypoint visibility (0-1)."""
        return self.count_visible(TORSO_KEYPOINTS) / len(TORSO_KEYPOINTS)

    @property
    def arm_quality(self) -> float:
        """Quality score based on arm keypoint visibility (0-1)."""
        return self.count_visible(ARM_KEYPOINTS) / len(ARM_KEYPOINTS)


class PoseEstimator(ABC):
    """Abstract base class for pose estimation."""

    @abstractmethod
    def estimate(self, frame_bgr) -> Optional[PoseResult]:
        """
        Estimate pose from a BGR frame.

        Args:
            frame_bgr: OpenCV BGR image (numpy array)

        Returns:
            PoseResult if a pose is detected, None otherwise
        """
        pass

    @abstractmethod
    def close(self):
        """Release any resources held by the estimator."""
        pass

    def get_skeleton_connections(self) -> List[tuple]:
        """
        Return pairs of keypoint names that should be connected when drawing.

        Returns:
            List of (KeypointName, KeypointName) tuples
        """
        return [
            # Head
            (KeypointName.NOSE, KeypointName.LEFT_EYE),
            (KeypointName.NOSE, KeypointName.RIGHT_EYE),
            (KeypointName.LEFT_EYE, KeypointName.LEFT_EAR),
            (KeypointName.RIGHT_EYE, KeypointName.RIGHT_EAR),
            # Torso
            (KeypointName.LEFT_SHOULDER, KeypointName.RIGHT_SHOULDER),
            (KeypointName.LEFT_SHOULDER, KeypointName.LEFT_HIP),
            (KeypointName.RIGHT_SHOULDER, KeypointName.RIGHT_HIP),
            (KeypointName.LEFT_HIP, KeypointName.RIGHT_HIP),
            # Arms
            (KeypointName.LEFT_SHOULDER, KeypointName.LEFT_ELBOW),
            (KeypointName.LEFT_ELBOW, KeypointName.LEFT_WRIST),
            (KeypointName.RIGHT_SHOULDER, KeypointName.RIGHT_ELBOW),
            (KeypointName.RIGHT_ELBOW, KeypointName.RIGHT_WRIST),
            # Legs
            (KeypointName.LEFT_HIP, KeypointName.LEFT_KNEE),
            (KeypointName.LEFT_KNEE, KeypointName.LEFT_ANKLE),
            (KeypointName.RIGHT_HIP, KeypointName.RIGHT_KNEE),
            (KeypointName.RIGHT_KNEE, KeypointName.RIGHT_ANKLE),
        ]
