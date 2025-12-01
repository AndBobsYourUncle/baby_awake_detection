"""
Debug Visualizer - Overlay rendering and CSV logging for debugging.

Provides:
1. Skeleton visualization with torso/arm highlighting
2. State and metrics overlay
3. CSV logging for offline analysis
"""

import csv
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .pose_api import (
    PoseResult,
    PoseEstimator,
    KeypointName,
    TORSO_KEYPOINTS,
    ARM_KEYPOINTS,
)
from .motion_analyzer import MotionState
from .sleep_state_machine import SleepState, StateInfo
from .pose_classifier import PoseClass, AWAKE_POSES
from .config import DebugConfig


class DebugVisualizer:
    """
    Renders debug overlays and manages CSV logging.
    """

    # State colors (BGR)
    STATE_COLORS = {
        SleepState.EMPTY: (128, 128, 128),  # Gray
        SleepState.PRESENT: (255, 165, 0),  # Orange
        SleepState.ASLEEP: (255, 0, 0),     # Blue
        SleepState.AWAKE: (0, 255, 0),      # Green
    }

    def __init__(self, config: DebugConfig, estimator: Optional[PoseEstimator] = None):
        """
        Initialize debug visualizer.

        Args:
            config: Debug configuration
            estimator: Pose estimator (for skeleton connections)
        """
        self._config = config
        self._estimator = estimator
        self._skeleton_connections = estimator.get_skeleton_connections() if estimator else []

        # CSV logging
        self._csv_file = None
        self._csv_writer = None
        self._last_log_time = 0.0

        if config.enable_csv_logging:
            self._init_csv_logging()

    def _init_csv_logging(self):
        """Initialize CSV file for logging."""
        path = Path(self._config.csv_log_path)

        # Add timestamp to filename if file exists
        if path.exists():
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            path = path.with_stem(f"{path.stem}_{timestamp}")

        self._csv_file = open(path, 'w', newline='')
        self._csv_writer = csv.writer(self._csv_file)

        # Write header
        self._csv_writer.writerow([
            'timestamp',
            'state',
            'pose_quality',
            'pose_valid',
            'pose_class',
            'pose_class_confidence',
            'is_awake_pose',
            'movement_torso',
            'movement_arms',
            'movement_score',
            'smoothed_movement',
            'baby_present',
            'time_in_state',
            'transition_target',
            'transition_progress',
            'global_motion_x',
            'global_motion_y',
            'num_torso_visible',
            'num_arms_visible',
            'reason',
        ])

    def draw_overlay(
        self,
        frame: np.ndarray,
        pose: Optional[PoseResult],
        motion: MotionState,
        state_info: StateInfo,
        fps: float = 0.0,
    ) -> np.ndarray:
        """
        Draw debug overlay on frame.

        Args:
            frame: BGR image to draw on
            pose: Current pose result (or None)
            motion: Current motion state
            state_info: Current state machine info
            fps: Current FPS

        Returns:
            Frame with overlay drawn
        """
        if not self._config.show_overlay:
            return frame

        display = frame.copy()
        h, w = display.shape[:2]

        # Draw skeleton
        if self._config.show_skeleton and pose:
            display = self._draw_skeleton(display, pose)

        # Draw state box (top-left)
        color = self.STATE_COLORS.get(state_info.state, (128, 128, 128))
        cv2.rectangle(display, (10, 10), (200, 80), color, -1)
        cv2.rectangle(display, (10, 10), (200, 80), (255, 255, 255), 2)
        cv2.putText(
            display,
            state_info.state.value.upper(),
            (20, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )

        # Draw transition progress bar
        if state_info.transition_target and state_info.transition_progress > 0:
            bar_x, bar_y = 10, 85
            bar_width, bar_height = 180, 15
            progress = int(bar_width * state_info.transition_progress)

            # Background
            cv2.rectangle(
                display,
                (bar_x, bar_y),
                (bar_x + bar_width, bar_y + bar_height),
                (50, 50, 50),
                -1,
            )
            # Progress
            target_color = self.STATE_COLORS.get(state_info.transition_target, (200, 200, 200))
            cv2.rectangle(
                display,
                (bar_x, bar_y),
                (bar_x + progress, bar_y + bar_height),
                target_color,
                -1,
            )
            # Border
            cv2.rectangle(
                display,
                (bar_x, bar_y),
                (bar_x + bar_width, bar_y + bar_height),
                (255, 255, 255),
                1,
            )
            # Label
            cv2.putText(
                display,
                f"-> {state_info.transition_target.value}",
                (bar_x + bar_width + 5, bar_y + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )

        # Draw metrics panel (left side)
        y = 120

        # Format pose class name
        pose_class_str = motion.pose_class.name.replace("_", " ").title()
        awake_indicator = " [AWAKE]" if motion.is_awake_pose else ""

        metrics = [
            f"FPS: {fps:.1f}",
            f"Pose Quality: {motion.pose_quality*100:.0f}%",
            f"Torso: {motion.num_torso_visible}/{len(TORSO_KEYPOINTS)}",
            f"Arms: {motion.num_arms_visible}/{len(ARM_KEYPOINTS)}",
            "",
            f"Pose: {pose_class_str}{awake_indicator}",
            f"Pose Conf: {motion.pose_class_confidence:.2f}",
            "",
            f"Move (torso): {motion.movement_torso:.2f}",
            f"Move (arms): {motion.movement_arms:.2f}",
            f"Move (combined): {motion.movement_score:.2f}",
            f"Move (smoothed): {motion.smoothed_movement:.2f}",
            "",
            f"In state: {state_info.time_in_state:.1f}s",
            f"Confidence: {state_info.confidence:.2f}",
        ]

        for text in metrics:
            if text:
                cv2.putText(
                    display,
                    text,
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                )
            y += 18

        # Draw reason at bottom
        y += 10
        cv2.putText(
            display,
            f"Reason: {state_info.reason}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 200, 200),
            1,
        )

        # Draw global motion vector (if significant)
        gm_x, gm_y = motion.global_motion
        if abs(gm_x) > 1 or abs(gm_y) > 1:
            center = (w - 50, 50)
            end = (int(center[0] + gm_x * 2), int(center[1] + gm_y * 2))
            cv2.arrowedLine(display, center, end, (0, 255, 255), 2)
            cv2.putText(
                display,
                "Global",
                (w - 70, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (0, 255, 255),
                1,
            )

        return display

    def _draw_skeleton(self, frame: np.ndarray, pose: PoseResult) -> np.ndarray:
        """Draw skeleton with torso/arm highlighting."""
        display = frame.copy()
        min_conf = 0.3

        # Get colors
        torso_color = self._config.torso_color
        arm_color = self._config.arm_color if self._config.highlight_arms else torso_color
        low_conf_color = self._config.low_confidence_color

        # Draw connections
        for kp1_name, kp2_name in self._skeleton_connections:
            kp1 = pose.get(kp1_name)
            kp2 = pose.get(kp2_name)

            if kp1 and kp2 and kp1.confidence > min_conf and kp2.confidence > min_conf:
                pt1 = (int(kp1.x), int(kp1.y))
                pt2 = (int(kp2.x), int(kp2.y))

                # Determine color based on whether this is an arm connection
                is_arm = kp1_name in ARM_KEYPOINTS or kp2_name in ARM_KEYPOINTS
                color = arm_color if is_arm else torso_color

                cv2.line(display, pt1, pt2, color, 2)

        # Draw keypoints
        for kp_name, kp in pose.keypoints.items():
            if kp.confidence > min_conf:
                pt = (int(kp.x), int(kp.y))

                # Determine color
                if kp.confidence < 0.5:
                    color = low_conf_color
                elif kp_name in ARM_KEYPOINTS:
                    color = arm_color
                else:
                    color = torso_color

                # Draw circle
                radius = 6 if kp_name in ARM_KEYPOINTS else 5
                cv2.circle(display, pt, radius, color, -1)
                cv2.circle(display, pt, radius, (255, 255, 255), 1)

                # Optionally draw keypoint name
                if self._config.show_keypoint_names:
                    cv2.putText(
                        display,
                        kp_name.value[:3],
                        (pt[0] + 8, pt[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        (255, 255, 255),
                        1,
                    )

                # Optionally draw confidence
                if self._config.show_confidence:
                    cv2.putText(
                        display,
                        f"{kp.confidence:.1f}",
                        (pt[0] + 8, pt[1] + 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.25,
                        (200, 200, 200),
                        1,
                    )

        return display

    def log_frame(
        self,
        motion: MotionState,
        state_info: StateInfo,
        timestamp: Optional[float] = None,
    ):
        """
        Log frame data to CSV.

        Args:
            motion: Current motion state
            state_info: Current state machine info
            timestamp: Current timestamp
        """
        if not self._config.enable_csv_logging or not self._csv_writer:
            return

        if timestamp is None:
            timestamp = time.time()

        # Check log interval
        if self._config.log_interval_seconds > 0:
            if timestamp - self._last_log_time < self._config.log_interval_seconds:
                return

        self._last_log_time = timestamp

        self._csv_writer.writerow([
            timestamp,
            state_info.state.value,
            motion.pose_quality,
            motion.pose_valid,
            motion.pose_class.name,
            motion.pose_class_confidence,
            motion.is_awake_pose,
            motion.movement_torso,
            motion.movement_arms,
            motion.movement_score,
            motion.smoothed_movement,
            motion.baby_present,
            state_info.time_in_state,
            state_info.transition_target.value if state_info.transition_target else "",
            state_info.transition_progress,
            motion.global_motion[0],
            motion.global_motion[1],
            motion.num_torso_visible,
            motion.num_arms_visible,
            state_info.reason,
        ])

        # Flush periodically
        if self._csv_file:
            self._csv_file.flush()

    def close(self):
        """Close resources."""
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None
