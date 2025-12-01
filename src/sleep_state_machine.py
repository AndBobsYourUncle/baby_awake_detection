"""
Sleep State Machine - Hysteresis-based state transitions with arm-aware logic.

States:
- EMPTY: No baby detected in crib
- PRESENT: Baby detected but state uncertain
- ASLEEP: Baby detected, lying still
- AWAKE: Baby detected, moving (especially arms/hands)

Transitions use amplitude + time thresholds with proper hysteresis
to prevent flickering between states.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import time


class SleepState(Enum):
    """Possible sleep states."""
    EMPTY = "empty"
    PRESENT = "present"
    ASLEEP = "asleep"
    AWAKE = "awake"


@dataclass
class StateInfo:
    """Current state machine status."""
    state: SleepState
    confidence: float  # 0-1 confidence in current state

    # Timing info
    time_in_state: float  # Seconds in current state
    time_since_last_transition: float

    # Transition progress (for UI)
    transition_target: Optional[SleepState]  # State we're transitioning toward
    transition_progress: float  # 0-1 progress toward transition

    # Debug info
    reason: str  # Human-readable reason for current state


@dataclass
class StateThresholds:
    """Thresholds for state transitions."""
    # Movement thresholds
    movement_low: float = 0.1  # Below this = "still"
    movement_high: float = 0.3  # Above this = "moving"
    movement_arms_high: float = 0.4  # Arm-specific threshold for strong awake signal

    # Time thresholds (seconds)
    present_confirm_seconds: float = 2.0  # Good pose for this long = confirmed present
    still_to_asleep_seconds: float = 60.0  # Still for this long = asleep
    movement_to_awake_seconds: float = 5.0  # Moving for this long = awake
    no_detection_to_empty_seconds: float = 30.0  # No pose for this long = empty
    awake_pose_confirm_seconds: float = 1.0  # Awake pose (sitting, etc.) for this long = awake

    # Hysteresis
    hysteresis_seconds: float = 2.0  # Minimum time between state changes


class SleepStateMachine:
    """
    Manages sleep state transitions with hysteresis.

    Key design principles:
    1. Amplitude thresholds: movement_low and movement_high create a dead zone
    2. Time thresholds: require sustained evidence before transitioning
    3. Arm-aware: arm movement is a strong signal for AWAKE state
    4. Hysteresis: minimum time between transitions prevents flickering
    """

    def __init__(self, thresholds: Optional[StateThresholds] = None):
        """
        Initialize state machine.

        Args:
            thresholds: Custom thresholds (uses defaults if None)
        """
        self._thresholds = thresholds or StateThresholds()

        # Current state
        self._state = SleepState.EMPTY
        self._state_entry_time: Optional[float] = None
        self._last_transition_time: Optional[float] = None

        # Transition timers
        self._time_with_good_pose: float = 0.0
        self._time_below_low: float = 0.0
        self._time_above_high: float = 0.0
        self._time_arms_above_high: float = 0.0
        self._time_without_detection: float = 0.0
        self._time_in_awake_pose: float = 0.0  # Time in awake-indicating pose

        # Last update time for delta calculation
        self._last_update_time: Optional[float] = None

        # Pending transition info
        self._transition_target: Optional[SleepState] = None
        self._transition_start_time: Optional[float] = None

    def update(
        self,
        baby_present: bool,
        smoothed_movement: float,
        arm_movement: float,
        seconds_since_detection: float,
        is_awake_pose: bool = False,
        timestamp: Optional[float] = None,
    ) -> StateInfo:
        """
        Update state machine with new motion data.

        Args:
            baby_present: Whether baby pose is currently detected (with quality)
            smoothed_movement: Smoothed combined movement score (0-1)
            arm_movement: Smoothed arm-specific movement score (0-1)
            seconds_since_detection: Time since last valid pose detection
            is_awake_pose: True if pose classification indicates awake (sitting, on elbows, etc.)
            timestamp: Current timestamp (uses time.time() if not provided)

        Returns:
            StateInfo with current state and debug info
        """
        if timestamp is None:
            timestamp = time.time()

        # Calculate time delta
        dt = 0.0
        if self._last_update_time is not None:
            dt = timestamp - self._last_update_time
        self._last_update_time = timestamp

        # Initialize state entry time if needed
        if self._state_entry_time is None:
            self._state_entry_time = timestamp
            self._last_transition_time = timestamp

        # Update timers based on current conditions
        self._update_timers(
            baby_present=baby_present,
            smoothed_movement=smoothed_movement,
            arm_movement=arm_movement,
            seconds_since_detection=seconds_since_detection,
            is_awake_pose=is_awake_pose,
            dt=dt,
        )

        # Determine potential state transition
        new_state, reason = self._evaluate_transition(
            baby_present=baby_present,
            smoothed_movement=smoothed_movement,
            arm_movement=arm_movement,
            is_awake_pose=is_awake_pose,
            timestamp=timestamp,
        )

        # Apply hysteresis check
        if new_state != self._state:
            time_since_last = timestamp - (self._last_transition_time or timestamp)
            if time_since_last < self._thresholds.hysteresis_seconds:
                # Too soon for another transition
                new_state = self._state
                reason = f"Hysteresis ({time_since_last:.1f}s < {self._thresholds.hysteresis_seconds}s)"

        # Execute transition if needed
        if new_state != self._state:
            self._execute_transition(new_state, timestamp)

        # Calculate transition progress for UI
        transition_target, transition_progress = self._get_transition_progress(timestamp)

        return StateInfo(
            state=self._state,
            confidence=self._calculate_confidence(smoothed_movement),
            time_in_state=timestamp - self._state_entry_time,
            time_since_last_transition=timestamp - (self._last_transition_time or timestamp),
            transition_target=transition_target,
            transition_progress=transition_progress,
            reason=reason,
        )

    def _update_timers(
        self,
        baby_present: bool,
        smoothed_movement: float,
        arm_movement: float,
        seconds_since_detection: float,
        is_awake_pose: bool,
        dt: float,
    ):
        """Update internal timers based on current conditions."""
        # Timer: good pose detection
        if baby_present:
            self._time_with_good_pose += dt
            self._time_without_detection = 0.0
        else:
            self._time_with_good_pose = 0.0
            self._time_without_detection = seconds_since_detection

        # Timer: movement below low threshold (stillness)
        # Reset if in awake pose - awake poses override stillness
        if smoothed_movement < self._thresholds.movement_low and not is_awake_pose:
            self._time_below_low += dt
        else:
            self._time_below_low = 0.0

        # Timer: movement above high threshold
        if smoothed_movement > self._thresholds.movement_high:
            self._time_above_high += dt
        else:
            self._time_above_high = 0.0

        # Timer: arm movement above arm-specific threshold
        if arm_movement > self._thresholds.movement_arms_high:
            self._time_arms_above_high += dt
        else:
            self._time_arms_above_high = 0.0

        # Timer: time in awake-indicating pose (sitting, on elbows, etc.)
        if is_awake_pose:
            self._time_in_awake_pose += dt
        else:
            self._time_in_awake_pose = 0.0

    def _evaluate_transition(
        self,
        baby_present: bool,
        smoothed_movement: float,
        arm_movement: float,
        is_awake_pose: bool,
        timestamp: float,
    ) -> tuple:
        """
        Evaluate potential state transition.

        Returns:
            (new_state, reason) tuple
        """
        th = self._thresholds

        # From EMPTY state
        if self._state == SleepState.EMPTY:
            if self._time_with_good_pose >= th.present_confirm_seconds:
                # Baby detected long enough to confirm presence
                # Check awake indicators: movement, arm movement, or awake pose
                if is_awake_pose:
                    return SleepState.AWAKE, "Baby detected in awake pose"
                elif smoothed_movement > th.movement_high or arm_movement > th.movement_arms_high:
                    return SleepState.AWAKE, f"Baby detected + moving ({smoothed_movement:.2f})"
                else:
                    return SleepState.ASLEEP, f"Baby detected + still ({smoothed_movement:.2f})"
            return self._state, "Waiting for pose confirmation"

        # From PRESENT state
        if self._state == SleepState.PRESENT:
            if self._time_without_detection >= th.no_detection_to_empty_seconds:
                return SleepState.EMPTY, f"No detection for {self._time_without_detection:.0f}s"

            # Awake pose is immediate awake indicator (after brief confirmation)
            if self._time_in_awake_pose >= th.awake_pose_confirm_seconds:
                return SleepState.AWAKE, f"Awake pose for {self._time_in_awake_pose:.1f}s"

            if self._time_above_high >= th.movement_to_awake_seconds:
                return SleepState.AWAKE, f"Moving for {self._time_above_high:.0f}s"

            if self._time_arms_above_high >= th.movement_to_awake_seconds * 0.5:
                # Arm movement gets faster transition to awake
                return SleepState.AWAKE, f"Arms moving for {self._time_arms_above_high:.0f}s"

            if self._time_below_low >= th.still_to_asleep_seconds:
                return SleepState.ASLEEP, f"Still for {self._time_below_low:.0f}s"

            return self._state, "Determining state..."

        # From ASLEEP state
        if self._state == SleepState.ASLEEP:
            if self._time_without_detection >= th.no_detection_to_empty_seconds:
                return SleepState.EMPTY, f"No detection for {self._time_without_detection:.0f}s"

            # Awake pose immediately triggers awake (after brief confirmation)
            if self._time_in_awake_pose >= th.awake_pose_confirm_seconds:
                return SleepState.AWAKE, f"Awake pose for {self._time_in_awake_pose:.1f}s"

            # Check for awake transition via movement
            movement_awake = self._time_above_high >= th.movement_to_awake_seconds
            arms_awake = self._time_arms_above_high >= th.movement_to_awake_seconds * 0.5

            if movement_awake or arms_awake:
                if arms_awake:
                    return SleepState.AWAKE, f"Arms moving for {self._time_arms_above_high:.0f}s"
                return SleepState.AWAKE, f"Moving for {self._time_above_high:.0f}s"

            return self._state, f"Asleep (still for {self._time_below_low:.0f}s)"

        # From AWAKE state
        if self._state == SleepState.AWAKE:
            if self._time_without_detection >= th.no_detection_to_empty_seconds:
                return SleepState.EMPTY, f"No detection for {self._time_without_detection:.0f}s"

            # Can only transition to asleep if NOT in awake pose
            if self._time_below_low >= th.still_to_asleep_seconds and not is_awake_pose:
                return SleepState.ASLEEP, f"Still for {self._time_below_low:.0f}s"

            # Stay awake if in awake pose
            if is_awake_pose:
                return self._state, f"Awake (pose: awake position)"

            return self._state, f"Awake (moving: {smoothed_movement:.2f})"

        return self._state, "Unknown state"

    def _execute_transition(self, new_state: SleepState, timestamp: float):
        """Execute a state transition."""
        self._state = new_state
        self._state_entry_time = timestamp
        self._last_transition_time = timestamp

        # Reset relevant timers based on new state
        if new_state == SleepState.EMPTY:
            self._time_with_good_pose = 0.0
        elif new_state == SleepState.ASLEEP:
            self._time_above_high = 0.0
            self._time_arms_above_high = 0.0
        elif new_state == SleepState.AWAKE:
            self._time_below_low = 0.0

    def _get_transition_progress(self, timestamp: float) -> tuple:
        """
        Get pending transition info for UI display.

        Returns:
            (target_state, progress_0_to_1) or (None, 0) if no pending transition
        """
        th = self._thresholds

        if self._state == SleepState.ASLEEP:
            # Check progress toward AWAKE
            if self._time_above_high > 0 or self._time_arms_above_high > 0:
                progress = max(
                    self._time_above_high / th.movement_to_awake_seconds,
                    self._time_arms_above_high / (th.movement_to_awake_seconds * 0.5),
                )
                return SleepState.AWAKE, min(progress, 1.0)

        elif self._state == SleepState.AWAKE:
            # Check progress toward ASLEEP
            if self._time_below_low > 0:
                progress = self._time_below_low / th.still_to_asleep_seconds
                return SleepState.ASLEEP, min(progress, 1.0)

        elif self._state == SleepState.EMPTY:
            # Check progress toward PRESENT/ASLEEP
            if self._time_with_good_pose > 0:
                progress = self._time_with_good_pose / th.present_confirm_seconds
                return SleepState.ASLEEP, min(progress, 1.0)

        # Check progress toward EMPTY (from any state)
        if self._time_without_detection > 0 and self._state != SleepState.EMPTY:
            progress = self._time_without_detection / th.no_detection_to_empty_seconds
            if progress > 0.1:  # Only show if meaningful progress
                return SleepState.EMPTY, min(progress, 1.0)

        return None, 0.0

    def _calculate_confidence(self, smoothed_movement: float) -> float:
        """Calculate confidence in current state."""
        th = self._thresholds

        if self._state == SleepState.ASLEEP:
            # Higher confidence when movement is well below threshold
            if smoothed_movement < th.movement_low:
                return min(1.0, (th.movement_low - smoothed_movement) / th.movement_low + 0.5)
            return 0.5

        elif self._state == SleepState.AWAKE:
            # Higher confidence when movement is well above threshold
            if smoothed_movement > th.movement_high:
                return min(1.0, (smoothed_movement - th.movement_high) / (1.0 - th.movement_high) + 0.5)
            return 0.5

        elif self._state == SleepState.EMPTY:
            return 1.0 if self._time_without_detection > th.no_detection_to_empty_seconds else 0.5

        return 0.5

    def reset(self):
        """Reset state machine to initial state."""
        self._state = SleepState.EMPTY
        self._state_entry_time = None
        self._last_transition_time = None
        self._time_with_good_pose = 0.0
        self._time_below_low = 0.0
        self._time_above_high = 0.0
        self._time_arms_above_high = 0.0
        self._time_without_detection = 0.0
        self._time_in_awake_pose = 0.0
        self._last_update_time = None

    @property
    def state(self) -> SleepState:
        """Get current state."""
        return self._state


# Re-export for convenience
__all__ = ['SleepStateMachine', 'SleepState', 'StateInfo', 'StateThresholds']
