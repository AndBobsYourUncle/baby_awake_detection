"""
Sleep Classifier - Coordinates detection and state management.

This is the main interface that ties together:
1. PoseDetector - raw frame detection
2. DetectionAccumulator - smoothing and dropout handling
3. SleepStateMachine - hysteresis-based state transitions
"""

from dataclasses import dataclass
from typing import Optional
import time

import numpy as np

from .pose_detector import PoseDetector, DetectionResult
from .detection_accumulator import DetectionAccumulator, AccumulatedState
from .state_machine import SleepStateMachine, SleepState, StateInfo, TransitionThresholds


@dataclass
class ClassificationResult:
    """Complete classification result for a frame."""
    state: SleepState
    confidence: float

    # Detection info
    has_detection: bool
    detection_rate: float  # What % of recent frames had valid detections
    landmarks: Optional[np.ndarray]
    visible_landmarks: int  # How many landmarks visible this frame

    # Movement info
    movement_score: float

    # State machine info
    time_in_state: float
    transition_progress: Optional[float]  # Progress toward next state (0-1)
    transition_target: Optional[SleepState]

    # Debug info
    reason: str


class SleepClassifier:
    """
    Main classifier that coordinates pose detection and sleep state tracking.

    Usage:
        classifier = SleepClassifier()
        result = classifier.process_frame(frame)
        print(f"State: {result.state.value}")
    """

    def __init__(
        self,
        # Detection settings
        min_detection_confidence: float = 0.3,
        model_complexity: int = 1,
        enable_ir_preprocessing: bool = True,

        # Accumulator settings
        buffer_seconds: float = 3.0,
        smoothing_factor: float = 0.3,

        # State machine thresholds
        thresholds: Optional[TransitionThresholds] = None,
    ):
        """
        Initialize the sleep classifier.

        Args:
            min_detection_confidence: MediaPipe detection confidence (lower = more detections)
            model_complexity: MediaPipe model (0=lite, 1=full, 2=heavy)
            enable_ir_preprocessing: Apply CLAHE for IR cameras
            buffer_seconds: How many seconds of detections to buffer
            smoothing_factor: Landmark smoothing (0=none, 1=full)
            thresholds: Custom state transition thresholds
        """
        self._detector = PoseDetector(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_detection_confidence,
            model_complexity=model_complexity,
            enable_ir_preprocessing=enable_ir_preprocessing,
        )

        self._accumulator = DetectionAccumulator(
            buffer_seconds=buffer_seconds,
            smoothing_factor=smoothing_factor,
        )

        self._state_machine = SleepStateMachine(
            thresholds=thresholds or TransitionThresholds(),
        )

    def process_frame(self, frame: np.ndarray) -> ClassificationResult:
        """
        Process a single frame and return classification result.

        Args:
            frame: BGR image from camera

        Returns:
            ClassificationResult with state and debug info
        """
        timestamp = time.time()

        # 1. Detect pose in frame
        detection = self._detector.detect(frame)

        # 2. Update accumulator with detection
        accumulated = self._accumulator.update(
            landmarks=detection.landmarks if detection.detected else None,
            timestamp=timestamp,
        )

        # 3. Update state machine
        state_info = self._state_machine.update(
            has_detection=accumulated.has_detection,
            movement_score=accumulated.movement_score,
            seconds_since_detection=accumulated.seconds_since_last_detection,
            timestamp=timestamp,
        )

        return ClassificationResult(
            state=state_info.state,
            confidence=state_info.confidence,
            has_detection=accumulated.has_detection,
            detection_rate=accumulated.detection_rate,
            landmarks=accumulated.landmarks,
            visible_landmarks=detection.visible_landmarks if detection.detected else 0,
            movement_score=accumulated.movement_score,
            time_in_state=state_info.time_in_state,
            transition_progress=state_info.transition_progress,
            transition_target=state_info.transition_target,
            reason=state_info.reason,
        )

    def get_detector(self) -> PoseDetector:
        """Get the pose detector for drawing skeletons."""
        return self._detector

    def reset(self):
        """Reset all state."""
        self._accumulator.reset()
        self._state_machine.reset()

    def close(self):
        """Release resources."""
        self._detector.close()


# Re-export SleepState for convenience
__all__ = ['SleepClassifier', 'ClassificationResult', 'SleepState', 'TransitionThresholds']
