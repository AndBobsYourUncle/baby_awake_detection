"""
Pose Classifier - Rule-based classification of baby static postures.

This module classifies the baby's pose into categories that can help
determine sleep/awake state. Certain poses (sitting, on elbows, etc.)
strongly indicate awake even without movement.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, List

from .pose_features import PoseFeatures


class PoseClass(Enum):
    """Static pose classifications."""
    UNKNOWN = auto()        # Insufficient data to classify
    LYING_FLAT = auto()     # On back or stomach, stretched out
    CURLED_TUCKED = auto()  # Fetal position or curled up
    SIDE_LYING = auto()     # On side
    ON_ELBOWS = auto()      # Propped up on elbows (awake indicator)
    ON_ALL_FOURS = auto()   # Crawling position (awake indicator)
    SITTING = auto()        # Sitting up (awake indicator)
    STANDING = auto()       # Standing (awake indicator)


# Poses that strongly indicate awake state
AWAKE_POSES = {
    PoseClass.ON_ELBOWS,
    PoseClass.ON_ALL_FOURS,
    PoseClass.SITTING,
    PoseClass.STANDING,
}

# Poses that are neutral (could be asleep or awake)
NEUTRAL_POSES = {
    PoseClass.LYING_FLAT,
    PoseClass.CURLED_TUCKED,
    PoseClass.SIDE_LYING,
}


@dataclass
class ClassificationResult:
    """Result of pose classification."""
    pose_class: PoseClass
    confidence: float  # 0-1 confidence in classification
    is_awake_pose: bool  # True if this pose implies awake
    reason: str  # Human-readable explanation


@dataclass
class PoseClassifierConfig:
    """Configuration thresholds for pose classification."""
    # Aspect ratio thresholds
    lying_flat_min_aspect: float = 1.5  # Width > 1.5x height = lying
    sitting_max_aspect: float = 0.8  # Width < 0.8x height = upright

    # Compactness thresholds
    curled_min_compactness: float = 2.5  # High compactness = curled
    lying_max_compactness: float = 2.0  # Low compactness = stretched

    # Asymmetry thresholds
    side_lying_min_asymmetry: float = 0.15  # Significant left-right asymmetry

    # Elbow angle thresholds (degrees)
    on_elbows_max_angle: float = 120  # Bent elbows
    on_elbows_min_forward: float = 0.05  # Elbows forward of shoulders

    # Arm elevation (normalized, negative = raised)
    arms_raised_threshold: float = -0.02  # Wrists above shoulders

    # Head tilt
    upright_max_tilt: float = 30  # Degrees from vertical


def classify_pose(
    features: PoseFeatures,
    config: Optional[PoseClassifierConfig] = None,
) -> ClassificationResult:
    """
    Classify baby's static posture using rule-based heuristics.

    Args:
        features: Extracted pose features
        config: Classification thresholds

    Returns:
        ClassificationResult with pose class and confidence
    """
    if config is None:
        config = PoseClassifierConfig()

    # Check if we have enough features to classify
    if features.pose_quality < 0.3:
        return ClassificationResult(
            pose_class=PoseClass.UNKNOWN,
            confidence=0.0,
            is_awake_pose=False,
            reason="Insufficient pose quality",
        )

    # Collect votes for each pose class
    votes: dict = {pc: 0.0 for pc in PoseClass}
    reasons: List[str] = []

    # === Check for SITTING/STANDING (upright poses) ===
    if features.aspect_ratio is not None:
        if features.aspect_ratio < config.sitting_max_aspect:
            # Tall and narrow = upright
            votes[PoseClass.SITTING] += 2.0
            votes[PoseClass.STANDING] += 1.0
            reasons.append(f"Upright aspect ({features.aspect_ratio:.2f})")

    # Head position relative to hips can indicate upright
    if features.head_to_hips_distance is not None and features.torso_length is not None:
        if features.torso_length > 0:
            head_hips_ratio = features.head_to_hips_distance / features.torso_length
            if head_hips_ratio < 1.2:  # Head close to hips vertically
                votes[PoseClass.SITTING] += 1.5
                reasons.append("Head close to hips")

    # === Check for ON_ELBOWS ===
    elbow_indicators = 0

    # Bent elbows
    if features.left_elbow_angle is not None and features.left_elbow_angle < config.on_elbows_max_angle:
        elbow_indicators += 1
    if features.right_elbow_angle is not None and features.right_elbow_angle < config.on_elbows_max_angle:
        elbow_indicators += 1

    # Elbows forward
    if features.left_elbow_forward is not None and features.left_elbow_forward > config.on_elbows_min_forward:
        elbow_indicators += 1
    if features.right_elbow_forward is not None and features.right_elbow_forward > config.on_elbows_min_forward:
        elbow_indicators += 1

    # Arms elevated (wrists near or above shoulders)
    if features.left_arm_elevation is not None and features.left_arm_elevation < 0:
        elbow_indicators += 1
    if features.right_arm_elevation is not None and features.right_arm_elevation < 0:
        elbow_indicators += 1

    if elbow_indicators >= 3:
        votes[PoseClass.ON_ELBOWS] += 2.0 + elbow_indicators * 0.5
        reasons.append(f"Elbow position ({elbow_indicators} indicators)")

    # === Check for ON_ALL_FOURS ===
    if features.body_compactness is not None and features.arm_spread is not None:
        if features.body_compactness > 1.5 and features.arm_spread is not None:
            # Wide arm spread with compact body
            if features.torso_length and features.arm_spread > features.torso_length * 0.5:
                votes[PoseClass.ON_ALL_FOURS] += 2.0
                reasons.append("Wide limb spread")

    # === Check for SIDE_LYING ===
    if features.side_asymmetry is not None:
        if features.side_asymmetry > config.side_lying_min_asymmetry:
            votes[PoseClass.SIDE_LYING] += 2.0
            reasons.append(f"Side asymmetry ({features.side_asymmetry:.2f})")

    # === Check for CURLED_TUCKED ===
    if features.body_compactness is not None:
        if features.body_compactness > config.curled_min_compactness:
            votes[PoseClass.CURLED_TUCKED] += 2.0
            reasons.append(f"High compactness ({features.body_compactness:.2f})")

    # Wrists close to head/body
    if features.left_wrist_to_head is not None and features.torso_length is not None:
        if features.left_wrist_to_head < features.torso_length * 0.5:
            votes[PoseClass.CURLED_TUCKED] += 1.0
    if features.right_wrist_to_head is not None and features.torso_length is not None:
        if features.right_wrist_to_head < features.torso_length * 0.5:
            votes[PoseClass.CURLED_TUCKED] += 1.0

    # === Check for LYING_FLAT ===
    if features.aspect_ratio is not None:
        if features.aspect_ratio > config.lying_flat_min_aspect:
            votes[PoseClass.LYING_FLAT] += 2.0
            reasons.append(f"Lying aspect ({features.aspect_ratio:.2f})")

    if features.body_compactness is not None:
        if features.body_compactness < config.lying_max_compactness:
            votes[PoseClass.LYING_FLAT] += 1.5
            reasons.append("Low compactness (stretched)")

    # Large head to hips distance = stretched out
    if features.head_to_hips_distance is not None and features.torso_length is not None:
        if features.torso_length > 0:
            ratio = features.head_to_hips_distance / features.torso_length
            if ratio > 1.3:
                votes[PoseClass.LYING_FLAT] += 1.5

    # === Select highest voted class ===
    # Default to LYING_FLAT for neutral poses if no strong signals
    votes[PoseClass.LYING_FLAT] += 0.5  # Slight bias toward lying

    best_class = max(votes, key=lambda k: votes[k])
    best_score = votes[best_class]

    # Calculate confidence based on vote margin
    sorted_votes = sorted(votes.values(), reverse=True)
    if len(sorted_votes) >= 2 and sorted_votes[0] > 0:
        margin = (sorted_votes[0] - sorted_votes[1]) / sorted_votes[0]
        confidence = min(0.5 + margin * 0.5, 1.0)
    else:
        confidence = 0.5

    # If no strong votes, return UNKNOWN
    if best_score < 1.0:
        best_class = PoseClass.UNKNOWN
        confidence = 0.3

    reason_str = "; ".join(reasons) if reasons else "Default classification"

    return ClassificationResult(
        pose_class=best_class,
        confidence=confidence,
        is_awake_pose=best_class in AWAKE_POSES,
        reason=reason_str,
    )


def is_awake_pose(pose_class: PoseClass) -> bool:
    """Check if a pose class indicates awake state."""
    return pose_class in AWAKE_POSES


def is_neutral_pose(pose_class: PoseClass) -> bool:
    """Check if a pose class is neutral (could be asleep or awake)."""
    return pose_class in NEUTRAL_POSES
