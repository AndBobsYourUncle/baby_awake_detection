"""
Detection Accumulator - Handles dropout tolerance and signal smoothing.

This module maintains a stable estimate of the baby's pose even when
MediaPipe has frequent detection dropouts (common with IR cameras).
"""

from dataclasses import dataclass, field
from typing import Optional
from collections import deque
import time

import numpy as np


@dataclass
class AccumulatedState:
    """Current accumulated state from the detection buffer."""
    has_detection: bool  # Do we have a valid detection estimate?
    detection_rate: float  # What % of recent frames had detections (0-1)
    landmarks: Optional[np.ndarray]  # Smoothed landmarks (or None)
    movement_score: float  # Relative movement score (0-1)
    seconds_since_last_detection: float  # How long since we last saw a pose
    confidence: float  # Overall confidence in current estimate


class DetectionAccumulator:
    """
    Accumulates pose detections over time to handle dropouts gracefully.

    Key responsibilities:
    1. Maintain rolling buffer of recent detections
    2. Smooth landmarks using exponential moving average
    3. Track detection rate (what % of frames have valid detections)
    4. Calculate relative movement (filters out camera/crib motion)
    5. Provide stable "best estimate" even during brief dropouts
    """

    def __init__(
        self,
        buffer_seconds: float = 3.0,
        fps_estimate: float = 30.0,
        smoothing_factor: float = 0.3,
        movement_threshold_pixels: float = 50.0,
    ):
        self.buffer_size = int(buffer_seconds * fps_estimate)
        self.smoothing_factor = smoothing_factor  # For EMA: 0=no smoothing, 1=full smoothing
        self.movement_threshold = movement_threshold_pixels

        # Rolling buffer of (timestamp, landmarks_or_none)
        self._detection_buffer: deque = deque(maxlen=self.buffer_size)

        # Smoothed state
        self._smoothed_landmarks: Optional[np.ndarray] = None
        self._last_detection_time: Optional[float] = None

        # Movement tracking - use relative distances between body parts
        # This filters out camera shake and crib bounce
        self._prev_relative_distances: Optional[dict] = None
        self._movement_history: deque = deque(maxlen=30)  # ~1 second of movement scores

        # Key landmark indices for movement calculation (arms/hands - usually visible)
        self.NOSE = 0
        self.LEFT_SHOULDER = 11
        self.RIGHT_SHOULDER = 12
        self.LEFT_ELBOW = 13
        self.RIGHT_ELBOW = 14
        self.LEFT_WRIST = 15
        self.RIGHT_WRIST = 16

    def update(self, landmarks: Optional[np.ndarray], timestamp: Optional[float] = None) -> AccumulatedState:
        """
        Update accumulator with new detection (or None if no detection).

        Args:
            landmarks: Numpy array of shape (33, 4) with [x, y, z, visibility] per landmark,
                      or None if no detection this frame
            timestamp: Optional timestamp (uses time.time() if not provided)

        Returns:
            AccumulatedState with current best estimates
        """
        if timestamp is None:
            timestamp = time.time()

        # Add to buffer
        self._detection_buffer.append((timestamp, landmarks))

        # Update smoothed landmarks if we have a detection
        if landmarks is not None:
            self._update_smoothed_landmarks(landmarks)
            self._last_detection_time = timestamp

        # Calculate metrics
        detection_rate = self._calculate_detection_rate()
        movement_score = self._calculate_movement_score(landmarks)

        seconds_since_detection = 0.0
        if self._last_detection_time is not None:
            seconds_since_detection = timestamp - self._last_detection_time

        # Determine if we have a valid detection estimate
        # We have a detection if: recent detection OR we're holding a recent pose
        has_detection = (
            self._smoothed_landmarks is not None and
            seconds_since_detection < 5.0  # Hold for up to 5 seconds
        )

        # Calculate confidence based on detection rate and recency
        confidence = self._calculate_confidence(detection_rate, seconds_since_detection)

        return AccumulatedState(
            has_detection=has_detection,
            detection_rate=detection_rate,
            landmarks=self._smoothed_landmarks.copy() if self._smoothed_landmarks is not None else None,
            movement_score=movement_score,
            seconds_since_last_detection=seconds_since_detection,
            confidence=confidence,
        )

    def _update_smoothed_landmarks(self, new_landmarks: np.ndarray):
        """Update smoothed landmarks using exponential moving average."""
        if self._smoothed_landmarks is None:
            self._smoothed_landmarks = new_landmarks.copy()
            return

        alpha = self.smoothing_factor
        smoothed = self._smoothed_landmarks.copy()

        for i in range(len(new_landmarks)):
            # Only smooth if both have reasonable visibility
            if new_landmarks[i, 3] > 0.3 and smoothed[i, 3] > 0.3:
                # EMA on position
                smoothed[i, :3] = alpha * smoothed[i, :3] + (1 - alpha) * new_landmarks[i, :3]
            elif new_landmarks[i, 3] > 0.3:
                # New detection has visibility, old doesn't - use new
                smoothed[i] = new_landmarks[i]
            # If new has low visibility, keep old smoothed value

            # Always update visibility to current
            smoothed[i, 3] = max(smoothed[i, 3] * 0.9, new_landmarks[i, 3])

        self._smoothed_landmarks = smoothed

    def _calculate_detection_rate(self) -> float:
        """Calculate what fraction of recent frames had valid detections."""
        if not self._detection_buffer:
            return 0.0

        valid_count = sum(1 for _, lm in self._detection_buffer if lm is not None)
        return valid_count / len(self._detection_buffer)

    def _calculate_movement_score(self, landmarks: Optional[np.ndarray]) -> float:
        """
        Calculate movement based on relative distances between body parts.
        This filters out camera shake and crib bounce since those move all points equally.
        """
        if landmarks is None or self._smoothed_landmarks is None:
            # No new detection - movement is 0 (or maintain last value)
            if self._movement_history:
                return sum(self._movement_history) / len(self._movement_history)
            return 0.0

        # Calculate relative distances between key body parts
        current_distances = self._get_relative_distances(landmarks)

        if self._prev_relative_distances is None:
            self._prev_relative_distances = current_distances
            return 0.0

        # Calculate how much the relative distances changed
        changes = []
        for key in current_distances:
            if key in self._prev_relative_distances:
                curr = current_distances[key]
                prev = self._prev_relative_distances[key]
                if curr is not None and prev is not None:
                    changes.append(abs(curr - prev))

        self._prev_relative_distances = current_distances

        if not changes:
            return 0.0

        # Normalize movement score
        avg_change = np.mean(changes)
        movement = min(avg_change / self.movement_threshold, 1.0)

        # Add to history and return smoothed value
        self._movement_history.append(movement)
        return sum(self._movement_history) / len(self._movement_history)

    def _get_relative_distances(self, landmarks: np.ndarray) -> dict:
        """Get distances between key body part pairs."""
        def dist(i1, i2):
            if landmarks[i1, 3] > 0.3 and landmarks[i2, 3] > 0.3:
                return np.linalg.norm(landmarks[i1, :2] - landmarks[i2, :2])
            return None

        return {
            'nose_to_left_shoulder': dist(self.NOSE, self.LEFT_SHOULDER),
            'nose_to_right_shoulder': dist(self.NOSE, self.RIGHT_SHOULDER),
            'shoulder_width': dist(self.LEFT_SHOULDER, self.RIGHT_SHOULDER),
            'left_arm': dist(self.LEFT_SHOULDER, self.LEFT_WRIST),
            'right_arm': dist(self.RIGHT_SHOULDER, self.RIGHT_WRIST),
            'left_forearm': dist(self.LEFT_ELBOW, self.LEFT_WRIST),
            'right_forearm': dist(self.RIGHT_ELBOW, self.RIGHT_WRIST),
        }

    def _calculate_confidence(self, detection_rate: float, seconds_since: float) -> float:
        """Calculate overall confidence in current state estimate."""
        # Base confidence from detection rate
        rate_confidence = detection_rate

        # Decay confidence based on time since last detection
        time_decay = max(0, 1.0 - seconds_since / 5.0)

        return rate_confidence * time_decay

    def reset(self):
        """Reset accumulator state."""
        self._detection_buffer.clear()
        self._smoothed_landmarks = None
        self._last_detection_time = None
        self._prev_relative_distances = None
        self._movement_history.clear()
