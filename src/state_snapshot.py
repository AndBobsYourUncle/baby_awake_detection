"""
State Snapshot - Unified state representation for MQTT publishing.

This module provides a simple dataclass that captures all relevant
state information for publishing to MQTT / Home Assistant.
"""

from dataclasses import dataclass
from typing import Optional
import time

from .motion_analyzer import MotionState
from .sleep_state_machine import SleepState, StateInfo


@dataclass
class StateSnapshot:
    """Snapshot of current baby monitor state for MQTT publishing."""
    # Detection state
    present: bool  # Baby detected in crib
    awake: bool  # Baby is awake (state == AWAKE)

    # State machine
    state_name: str  # "EMPTY", "PRESENT", "ASLEEP", "AWAKE"

    # Movement metrics (0-1 range)
    movement_torso: float
    movement_arms: float
    movement_combined: float
    movement_smoothed: float

    # Pose quality (0-100 percentage)
    pose_quality: float

    # Pose classification
    pose_class_name: str  # e.g., "LYING_FLAT", "SITTING", etc.

    # Timestamp
    timestamp: float

    @classmethod
    def from_state(
        cls,
        motion: MotionState,
        state_info: StateInfo,
        timestamp: Optional[float] = None,
    ) -> "StateSnapshot":
        """
        Create a StateSnapshot from motion and state machine outputs.

        Args:
            motion: Current MotionState from motion analyzer
            state_info: Current StateInfo from state machine
            timestamp: Optional timestamp (uses time.time() if not provided)

        Returns:
            StateSnapshot with all current values
        """
        if timestamp is None:
            timestamp = time.time()

        return cls(
            present=motion.baby_present,
            awake=state_info.state == SleepState.AWAKE,
            state_name=state_info.state.value,  # Already lowercase from enum
            movement_torso=motion.movement_torso,
            movement_arms=motion.movement_arms,
            movement_combined=motion.movement_score,
            movement_smoothed=motion.smoothed_movement,
            pose_quality=motion.pose_quality * 100.0,  # Convert to percentage
            pose_class_name=motion.pose_class.name.lower(),  # Lowercase for HA
            timestamp=timestamp,
        )
