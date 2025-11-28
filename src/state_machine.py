"""
State Machine - Hysteresis-based sleep state transitions.

States:
- EMPTY: No baby detected in crib
- ASLEEP: Baby is sleeping (lying still)
- AWAKE: Baby is awake (moving, sitting up, or eyes open)

Key principle: Each transition requires sustained evidence to prevent flickering.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import time


class SleepState(Enum):
    EMPTY = "empty"      # No baby in crib
    ASLEEP = "asleep"    # Baby sleeping
    AWAKE = "awake"      # Baby awake


@dataclass
class TransitionThresholds:
    """Thresholds for state transitions (in seconds)."""
    # EMPTY -> ASLEEP/AWAKE
    detection_to_present: float = 2.0  # Need pose for 2s to confirm baby present

    # ASLEEP -> AWAKE
    movement_to_awake: float = 5.0  # Need sustained movement for 5s

    # AWAKE -> ASLEEP
    still_to_asleep: float = 60.0  # Need to be still for 60s

    # ANY -> EMPTY
    no_detection_to_empty: float = 30.0  # No detection for 30s = empty crib

    # Movement threshold (0-1 scale)
    movement_threshold: float = 0.3  # Movement score above this = "moving"


@dataclass
class StateInfo:
    """Information about current state."""
    state: SleepState
    confidence: float
    time_in_state: float  # Seconds in current state
    transition_progress: Optional[float]  # Progress toward next state (0-1) if transitioning
    transition_target: Optional[SleepState]  # What state we're transitioning toward
    reason: str  # Human-readable reason for current state


class SleepStateMachine:
    """
    Manages sleep state with hysteresis-based transitions.

    The key insight is that transitions require sustained evidence:
    - Brief movement during sleep doesn't trigger AWAKE (sleep tossing)
    - Brief detection dropouts don't trigger EMPTY
    - Brief stillness doesn't trigger ASLEEP
    """

    def __init__(self, thresholds: Optional[TransitionThresholds] = None):
        self.thresholds = thresholds or TransitionThresholds()

        self._current_state = SleepState.EMPTY
        self._state_start_time = time.time()

        # Transition tracking
        self._transition_start_time: Optional[float] = None
        self._transition_target: Optional[SleepState] = None

        # Condition tracking (how long conditions have been met)
        self._condition_start_times: dict[str, Optional[float]] = {
            'has_detection': None,
            'no_detection': None,
            'moving': None,
            'still': None,
        }

    def update(
        self,
        has_detection: bool,
        movement_score: float,
        seconds_since_detection: float,
        timestamp: Optional[float] = None,
    ) -> StateInfo:
        """
        Update state machine with current signals.

        Args:
            has_detection: Whether we have a valid pose detection
            movement_score: Movement level (0-1, higher = more movement)
            seconds_since_detection: Time since last valid detection
            timestamp: Current time (uses time.time() if not provided)

        Returns:
            StateInfo with current state and metadata
        """
        if timestamp is None:
            timestamp = time.time()

        # Update condition tracking
        is_moving = movement_score > self.thresholds.movement_threshold
        self._update_condition('has_detection', has_detection, timestamp)
        self._update_condition('no_detection', not has_detection, timestamp)
        self._update_condition('moving', is_moving and has_detection, timestamp)
        self._update_condition('still', not is_moving and has_detection, timestamp)

        # Check for state transitions based on current state
        new_state, reason = self._evaluate_transitions(timestamp, seconds_since_detection)

        if new_state != self._current_state:
            self._current_state = new_state
            self._state_start_time = timestamp
            self._reset_transition()

        # Calculate transition progress if we're heading somewhere
        transition_progress = None
        transition_target = None

        potential_target, progress = self._get_transition_progress(timestamp, seconds_since_detection)
        if potential_target and progress > 0:
            transition_progress = progress
            transition_target = potential_target

        time_in_state = timestamp - self._state_start_time

        return StateInfo(
            state=self._current_state,
            confidence=self._calculate_confidence(has_detection, movement_score),
            time_in_state=time_in_state,
            transition_progress=transition_progress,
            transition_target=transition_target,
            reason=reason,
        )

    def _update_condition(self, condition: str, is_met: bool, timestamp: float):
        """Track how long a condition has been continuously met."""
        if is_met:
            if self._condition_start_times[condition] is None:
                self._condition_start_times[condition] = timestamp
        else:
            self._condition_start_times[condition] = None

    def _condition_duration(self, condition: str, timestamp: float) -> float:
        """Get how long a condition has been continuously met."""
        start = self._condition_start_times[condition]
        if start is None:
            return 0.0
        return timestamp - start

    def _evaluate_transitions(self, timestamp: float, seconds_since_detection: float) -> tuple[SleepState, str]:
        """Evaluate if we should transition to a new state."""

        # Priority 1: Check for EMPTY transition (any state -> EMPTY)
        if seconds_since_detection >= self.thresholds.no_detection_to_empty:
            return SleepState.EMPTY, f"no detection for {seconds_since_detection:.0f}s"

        # State-specific transitions
        if self._current_state == SleepState.EMPTY:
            return self._evaluate_from_empty(timestamp)
        elif self._current_state == SleepState.ASLEEP:
            return self._evaluate_from_asleep(timestamp)
        elif self._current_state == SleepState.AWAKE:
            return self._evaluate_from_awake(timestamp)

        return self._current_state, "no change"

    def _evaluate_from_empty(self, timestamp: float) -> tuple[SleepState, str]:
        """Evaluate transitions from EMPTY state."""
        detection_duration = self._condition_duration('has_detection', timestamp)
        moving_duration = self._condition_duration('moving', timestamp)

        # Need detection for threshold time before transitioning
        if detection_duration >= self.thresholds.detection_to_present:
            if moving_duration >= self.thresholds.detection_to_present:
                return SleepState.AWAKE, f"baby detected, moving"
            else:
                return SleepState.ASLEEP, f"baby detected, still"

        return SleepState.EMPTY, "waiting for sustained detection"

    def _evaluate_from_asleep(self, timestamp: float) -> tuple[SleepState, str]:
        """Evaluate transitions from ASLEEP state."""
        moving_duration = self._condition_duration('moving', timestamp)

        # ASLEEP -> AWAKE: Need sustained movement
        if moving_duration >= self.thresholds.movement_to_awake:
            return SleepState.AWAKE, f"sustained movement for {moving_duration:.1f}s"

        return SleepState.ASLEEP, "sleeping (still or brief movement)"

    def _evaluate_from_awake(self, timestamp: float) -> tuple[SleepState, str]:
        """Evaluate transitions from AWAKE state."""
        still_duration = self._condition_duration('still', timestamp)

        # AWAKE -> ASLEEP: Need to be still for a long time
        if still_duration >= self.thresholds.still_to_asleep:
            return SleepState.ASLEEP, f"still for {still_duration:.0f}s"

        return SleepState.AWAKE, "awake (moving or recently active)"

    def _get_transition_progress(self, timestamp: float, seconds_since_detection: float) -> tuple[Optional[SleepState], float]:
        """Get progress toward potential next state transition."""

        # Check empty transition
        if seconds_since_detection > 0:
            progress = seconds_since_detection / self.thresholds.no_detection_to_empty
            if progress > 0.1:  # Only show if meaningful progress
                return SleepState.EMPTY, min(progress, 1.0)

        if self._current_state == SleepState.EMPTY:
            detection_duration = self._condition_duration('has_detection', timestamp)
            if detection_duration > 0:
                progress = detection_duration / self.thresholds.detection_to_present
                return SleepState.ASLEEP, min(progress, 1.0)

        elif self._current_state == SleepState.ASLEEP:
            moving_duration = self._condition_duration('moving', timestamp)
            if moving_duration > 0:
                progress = moving_duration / self.thresholds.movement_to_awake
                return SleepState.AWAKE, min(progress, 1.0)

        elif self._current_state == SleepState.AWAKE:
            still_duration = self._condition_duration('still', timestamp)
            if still_duration > 0:
                progress = still_duration / self.thresholds.still_to_asleep
                return SleepState.ASLEEP, min(progress, 1.0)

        return None, 0.0

    def _calculate_confidence(self, has_detection: bool, movement_score: float) -> float:
        """Calculate confidence in current state."""
        base_confidence = 0.8 if has_detection else 0.3

        # Higher confidence when signals clearly match state
        if self._current_state == SleepState.ASLEEP:
            if movement_score < 0.1:
                return min(base_confidence + 0.2, 1.0)
        elif self._current_state == SleepState.AWAKE:
            if movement_score > 0.5:
                return min(base_confidence + 0.2, 1.0)
        elif self._current_state == SleepState.EMPTY:
            if not has_detection:
                return 0.9

        return base_confidence

    def _reset_transition(self):
        """Reset transition tracking."""
        self._transition_start_time = None
        self._transition_target = None

    def reset(self):
        """Reset state machine to initial state."""
        self._current_state = SleepState.EMPTY
        self._state_start_time = time.time()
        self._reset_transition()
        for key in self._condition_start_times:
            self._condition_start_times[key] = None

    @property
    def current_state(self) -> SleepState:
        return self._current_state
