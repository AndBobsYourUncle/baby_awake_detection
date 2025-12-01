"""
Motion Analyzer - Crib-relative movement detection with arm/torso separation.

This module handles:
1. Pose quality assessment
2. Global motion removal (filters crib bounce / camera shake)
3. Separate torso vs arm/hand movement tracking
4. Temporal smoothing with configurable windows
5. Pose feature extraction and classification
"""

from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Tuple
import time

import numpy as np

from .pose_api import (
    PoseResult,
    Keypoint,
    KeypointName,
    TORSO_KEYPOINTS,
    ARM_KEYPOINTS,
)
from .config import MotionConfig
from .pose_features import PoseFeatures, extract_features
from .pose_classifier import (
    PoseClass,
    ClassificationResult,
    classify_pose,
    is_awake_pose,
)


@dataclass
class MotionState:
    """Current motion analysis state."""
    # Pose quality
    pose_quality: float  # 0-1, based on visible torso keypoints
    pose_valid: bool  # True if enough keypoints visible

    # Movement scores (all 0-1 range)
    movement_torso: float  # Raw torso movement
    movement_arms: float  # Raw arm/hand movement
    movement_score: float  # Combined weighted score
    smoothed_movement: float  # Temporally smoothed combined score

    # Detection state
    baby_present: bool  # Pose seen recently with good quality
    seconds_since_detection: float  # Time since last good pose

    # Pose classification
    pose_class: PoseClass  # Static pose classification
    pose_class_confidence: float  # Confidence in classification
    is_awake_pose: bool  # True if pose implies awake state
    pose_class_reason: str  # Explanation for classification

    # Pose features (for debugging/logging)
    pose_features: Optional[PoseFeatures]

    # Debug info
    global_motion: Tuple[float, float]  # (dx, dy) of global motion this frame
    num_torso_visible: int
    num_arms_visible: int


class MotionAnalyzer:
    """
    Analyzes pose sequences to detect crib-relative movement.

    Key features:
    1. Subtracts global motion (crib bounce) using torso as anchor
    2. Tracks torso and arm movement separately
    3. Arms weighted more heavily for awake detection
    4. Temporal smoothing to reduce noise
    """

    def __init__(self, config: MotionConfig, fps_estimate: float = 30.0):
        """
        Initialize motion analyzer.

        Args:
            config: Motion configuration
            fps_estimate: Expected FPS for buffer sizing
        """
        self._config = config
        self._fps = fps_estimate

        # Previous pose for delta calculation
        self._prev_pose: Optional[PoseResult] = None
        self._prev_positions: dict = {}  # KeypointName -> (x, y)

        # Last detection time
        self._last_detection_time: Optional[float] = None

        # Movement history for smoothing
        buffer_size = int(config.smoothing_window_seconds * fps_estimate)
        self._torso_history: deque = deque(maxlen=buffer_size)
        self._arm_history: deque = deque(maxlen=buffer_size)
        self._combined_history: deque = deque(maxlen=buffer_size)

        # Pose buffer for dropout handling
        self._pose_buffer_size = int(config.buffer_seconds * fps_estimate)
        self._pose_buffer: deque = deque(maxlen=self._pose_buffer_size)

    def update(self, pose: Optional[PoseResult], timestamp: Optional[float] = None) -> MotionState:
        """
        Update motion analysis with new pose.

        Args:
            pose: PoseResult from pose estimator, or None if no detection
            timestamp: Current timestamp (uses time.time() if not provided)

        Returns:
            MotionState with current analysis
        """
        if timestamp is None:
            timestamp = time.time()

        # Add to pose buffer
        self._pose_buffer.append((timestamp, pose))

        # Calculate pose quality
        pose_quality = 0.0
        num_torso_visible = 0
        num_arms_visible = 0

        if pose:
            num_torso_visible = pose.count_visible(TORSO_KEYPOINTS, self._config.min_keypoint_confidence)
            num_arms_visible = pose.count_visible(ARM_KEYPOINTS, self._config.min_keypoint_confidence)
            pose_quality = num_torso_visible / len(TORSO_KEYPOINTS)

        pose_valid = num_torso_visible >= self._config.min_torso_keypoints

        # Update detection time if pose is valid
        if pose_valid:
            self._last_detection_time = timestamp

        # Calculate time since last detection
        seconds_since = 0.0
        if self._last_detection_time is not None:
            seconds_since = timestamp - self._last_detection_time

        # Determine if baby is present (valid pose recently)
        baby_present = (
            self._last_detection_time is not None and
            seconds_since < self._config.buffer_seconds
        )

        # Calculate movement if we have valid pose and previous pose
        movement_torso = 0.0
        movement_arms = 0.0
        global_motion = (0.0, 0.0)

        if pose_valid and pose and self._prev_pose is not None:
            movement_torso, movement_arms, global_motion = self._calculate_movement(pose)

        # Update previous pose
        if pose_valid and pose:
            self._prev_pose = pose
            self._update_prev_positions(pose)

        # Calculate combined score
        movement_score = (
            self._config.torso_weight * movement_torso +
            self._config.arm_weight * movement_arms
        )

        # Add to history
        self._torso_history.append(movement_torso)
        self._arm_history.append(movement_arms)
        self._combined_history.append(movement_score)

        # Calculate smoothed movement
        smoothed_movement = self._calculate_smoothed()

        # Extract pose features and classify
        pose_features = None
        pose_classification = ClassificationResult(
            pose_class=PoseClass.UNKNOWN,
            confidence=0.0,
            is_awake_pose=False,
            reason="No pose detected",
        )

        if pose and pose_valid:
            pose_features = extract_features(pose, self._config.min_keypoint_confidence)
            pose_classification = classify_pose(pose_features)

        return MotionState(
            pose_quality=pose_quality,
            pose_valid=pose_valid,
            movement_torso=movement_torso,
            movement_arms=movement_arms,
            movement_score=movement_score,
            smoothed_movement=smoothed_movement,
            baby_present=baby_present,
            seconds_since_detection=seconds_since,
            pose_class=pose_classification.pose_class,
            pose_class_confidence=pose_classification.confidence,
            is_awake_pose=pose_classification.is_awake_pose,
            pose_class_reason=pose_classification.reason,
            pose_features=pose_features,
            global_motion=global_motion,
            num_torso_visible=num_torso_visible,
            num_arms_visible=num_arms_visible,
        )

    def _calculate_movement(self, pose: PoseResult) -> Tuple[float, float, Tuple[float, float]]:
        """
        Calculate crib-relative movement for torso and arms.

        Args:
            pose: Current pose

        Returns:
            (torso_movement, arm_movement, global_motion)
        """
        # Get current positions for visible keypoints
        current_positions = {}
        for kp_name in list(TORSO_KEYPOINTS) + list(ARM_KEYPOINTS):
            kp = pose.get(kp_name)
            if kp and kp.confidence >= self._config.min_keypoint_confidence:
                current_positions[kp_name] = (kp.x, kp.y)

        # Calculate global motion from torso (anchor)
        global_dx, global_dy = 0.0, 0.0
        torso_deltas = []

        for kp_name in TORSO_KEYPOINTS:
            if kp_name in current_positions and kp_name in self._prev_positions:
                curr = current_positions[kp_name]
                prev = self._prev_positions[kp_name]
                dx = curr[0] - prev[0]
                dy = curr[1] - prev[1]
                torso_deltas.append((dx, dy))

        if torso_deltas:
            global_dx = np.mean([d[0] for d in torso_deltas])
            global_dy = np.mean([d[1] for d in torso_deltas])

        global_motion = (global_dx, global_dy)

        # Calculate relative movement for torso (after removing global motion)
        torso_movements = []
        for kp_name in TORSO_KEYPOINTS:
            if kp_name in current_positions and kp_name in self._prev_positions:
                curr = current_positions[kp_name]
                prev = self._prev_positions[kp_name]
                # Raw delta
                dx = curr[0] - prev[0]
                dy = curr[1] - prev[1]
                # Subtract global motion
                rel_dx = dx - global_dx
                rel_dy = dy - global_dy
                # Magnitude
                magnitude = np.sqrt(rel_dx**2 + rel_dy**2)
                torso_movements.append(magnitude)

        # Calculate relative movement for arms
        arm_movements = []
        for kp_name in ARM_KEYPOINTS:
            if kp_name in current_positions and kp_name in self._prev_positions:
                curr = current_positions[kp_name]
                prev = self._prev_positions[kp_name]
                # Raw delta
                dx = curr[0] - prev[0]
                dy = curr[1] - prev[1]
                # Subtract global motion
                rel_dx = dx - global_dx
                rel_dy = dy - global_dy
                # Magnitude
                magnitude = np.sqrt(rel_dx**2 + rel_dy**2)
                arm_movements.append(magnitude)

        # Normalize movements to 0-1 range
        scale = self._config.movement_scale_pixels

        torso_movement = 0.0
        if torso_movements:
            avg_torso = np.mean(torso_movements)
            max_torso = np.max(torso_movements)
            # Combine average and max (catch both sustained and sudden movement)
            torso_movement = min((avg_torso * 0.6 + max_torso * 0.4) / scale, 1.0)

        arm_movement = 0.0
        if arm_movements:
            avg_arm = np.mean(arm_movements)
            max_arm = np.max(arm_movements)
            arm_movement = min((avg_arm * 0.6 + max_arm * 0.4) / scale, 1.0)

        return torso_movement, arm_movement, global_motion

    def _update_prev_positions(self, pose: PoseResult):
        """Update previous positions for next frame comparison."""
        self._prev_positions.clear()
        for kp_name in list(TORSO_KEYPOINTS) + list(ARM_KEYPOINTS):
            kp = pose.get(kp_name)
            if kp and kp.confidence >= self._config.min_keypoint_confidence:
                self._prev_positions[kp_name] = (kp.x, kp.y)

    def _calculate_smoothed(self) -> float:
        """Calculate smoothed movement using configured method."""
        if not self._combined_history:
            return 0.0

        if self._config.smoothing_type == "median":
            return float(np.median(self._combined_history))
        else:
            # EMA (exponential moving average)
            alpha = self._config.ema_alpha
            smoothed = self._combined_history[0]
            for val in list(self._combined_history)[1:]:
                smoothed = alpha * val + (1 - alpha) * smoothed
            return smoothed

    def get_smoothed_arm_movement(self) -> float:
        """Get smoothed arm movement for arm-specific awake detection."""
        if not self._arm_history:
            return 0.0

        if self._config.smoothing_type == "median":
            return float(np.median(self._arm_history))
        else:
            alpha = self._config.ema_alpha
            smoothed = self._arm_history[0]
            for val in list(self._arm_history)[1:]:
                smoothed = alpha * val + (1 - alpha) * smoothed
            return smoothed

    def reset(self):
        """Reset analyzer state."""
        self._prev_pose = None
        self._prev_positions.clear()
        self._last_detection_time = None
        self._torso_history.clear()
        self._arm_history.clear()
        self._combined_history.clear()
        self._pose_buffer.clear()
