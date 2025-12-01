"""
Pose Features - Geometric feature extraction from pose keypoints.

This module transforms raw ViTPose/MediaPipe keypoints into a set of
geometric features that can be used for pose classification and
movement analysis.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import math

import numpy as np

from .pose_api import (
    PoseResult,
    Keypoint,
    KeypointName,
)


@dataclass
class PoseFeatures:
    """Bundle of geometric features extracted from a pose."""

    # Core measurements
    torso_center: Optional[Tuple[float, float]]  # (x, y) center of torso
    torso_length: Optional[float]  # Distance from shoulders to hips
    head_to_hips_distance: Optional[float]  # Vertical extent

    # Body shape
    body_compactness: Optional[float]  # Ratio of bbox area to torso length squared
    aspect_ratio: Optional[float]  # Width / height of body bounding box
    bounding_box: Optional[Tuple[float, float, float, float]]  # (x, y, w, h)

    # Arm configuration
    left_elbow_angle: Optional[float]  # Angle at left elbow (degrees)
    right_elbow_angle: Optional[float]  # Angle at right elbow (degrees)
    left_wrist_to_head: Optional[float]  # Distance from left wrist to nose
    right_wrist_to_head: Optional[float]  # Distance from right wrist to nose
    arm_spread: Optional[float]  # Mean distance of wrists from torso center

    # Arm elevation (relative to shoulders)
    left_arm_elevation: Optional[float]  # Wrist y relative to shoulder y (negative = raised)
    right_arm_elevation: Optional[float]

    # Elbow position relative to shoulders
    left_elbow_forward: Optional[float]  # Elbow x relative to shoulder
    right_elbow_forward: Optional[float]

    # Asymmetry
    side_asymmetry: Optional[float]  # Left-right asymmetry (0 = symmetric)

    # Head position
    head_tilt: Optional[float]  # Angle of head relative to shoulders

    # Overall pose quality
    pose_quality: float  # 0-1 based on visible keypoints


def extract_features(pose: PoseResult, min_confidence: float = 0.3) -> PoseFeatures:
    """
    Extract geometric features from a PoseResult.

    Args:
        pose: PoseResult from pose estimator
        min_confidence: Minimum keypoint confidence to use

    Returns:
        PoseFeatures dataclass with all computed features
    """
    # Helper to get keypoint if confident enough
    def get_kp(name: KeypointName) -> Optional[Keypoint]:
        kp = pose.get(name)
        if kp and kp.confidence >= min_confidence:
            return kp
        return None

    # Helper to compute distance between two keypoints
    def dist(kp1: Optional[Keypoint], kp2: Optional[Keypoint]) -> Optional[float]:
        if kp1 and kp2:
            return math.sqrt((kp1.x - kp2.x) ** 2 + (kp1.y - kp2.y) ** 2)
        return None

    # Helper to compute angle at middle point (in degrees)
    def angle_at(kp1: Optional[Keypoint], kp_mid: Optional[Keypoint], kp2: Optional[Keypoint]) -> Optional[float]:
        if not (kp1 and kp_mid and kp2):
            return None
        # Vectors from mid to endpoints
        v1 = (kp1.x - kp_mid.x, kp1.y - kp_mid.y)
        v2 = (kp2.x - kp_mid.x, kp2.y - kp_mid.y)
        # Dot product and magnitudes
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
        if mag1 < 1e-6 or mag2 < 1e-6:
            return None
        cos_angle = max(-1, min(1, dot / (mag1 * mag2)))
        return math.degrees(math.acos(cos_angle))

    # Get key keypoints
    nose = get_kp(KeypointName.NOSE)
    left_shoulder = get_kp(KeypointName.LEFT_SHOULDER)
    right_shoulder = get_kp(KeypointName.RIGHT_SHOULDER)
    left_elbow = get_kp(KeypointName.LEFT_ELBOW)
    right_elbow = get_kp(KeypointName.RIGHT_ELBOW)
    left_wrist = get_kp(KeypointName.LEFT_WRIST)
    right_wrist = get_kp(KeypointName.RIGHT_WRIST)
    left_hip = get_kp(KeypointName.LEFT_HIP)
    right_hip = get_kp(KeypointName.RIGHT_HIP)
    left_knee = get_kp(KeypointName.LEFT_KNEE)
    right_knee = get_kp(KeypointName.RIGHT_KNEE)

    # === Torso center ===
    torso_center = None
    torso_points = [kp for kp in [left_shoulder, right_shoulder, left_hip, right_hip] if kp]
    if len(torso_points) >= 2:
        cx = sum(kp.x for kp in torso_points) / len(torso_points)
        cy = sum(kp.y for kp in torso_points) / len(torso_points)
        torso_center = (cx, cy)

    # === Torso length ===
    torso_length = None
    shoulder_center = None
    hip_center = None
    if left_shoulder and right_shoulder:
        shoulder_center = ((left_shoulder.x + right_shoulder.x) / 2,
                           (left_shoulder.y + right_shoulder.y) / 2)
    if left_hip and right_hip:
        hip_center = ((left_hip.x + right_hip.x) / 2,
                      (left_hip.y + right_hip.y) / 2)
    if shoulder_center and hip_center:
        torso_length = math.sqrt((shoulder_center[0] - hip_center[0]) ** 2 +
                                  (shoulder_center[1] - hip_center[1]) ** 2)

    # === Head to hips distance ===
    head_to_hips_distance = None
    if nose and hip_center:
        head_to_hips_distance = math.sqrt((nose.x - hip_center[0]) ** 2 +
                                           (nose.y - hip_center[1]) ** 2)

    # === Bounding box and aspect ratio ===
    all_kps = [kp for kp in [nose, left_shoulder, right_shoulder,
                             left_elbow, right_elbow, left_wrist, right_wrist,
                             left_hip, right_hip, left_knee, right_knee] if kp]
    bounding_box = None
    aspect_ratio = None
    body_compactness = None

    if len(all_kps) >= 4:
        min_x = min(kp.x for kp in all_kps)
        max_x = max(kp.x for kp in all_kps)
        min_y = min(kp.y for kp in all_kps)
        max_y = max(kp.y for kp in all_kps)
        w = max_x - min_x
        h = max_y - min_y
        if h > 1:
            bounding_box = (min_x, min_y, w, h)
            aspect_ratio = w / h
            if torso_length and torso_length > 1:
                bbox_area = w * h
                body_compactness = bbox_area / (torso_length ** 2)

    # === Elbow angles ===
    left_elbow_angle = angle_at(left_shoulder, left_elbow, left_wrist)
    right_elbow_angle = angle_at(right_shoulder, right_elbow, right_wrist)

    # === Wrist to head distances ===
    left_wrist_to_head = dist(left_wrist, nose)
    right_wrist_to_head = dist(right_wrist, nose)

    # === Arm spread ===
    arm_spread = None
    if torso_center:
        wrist_dists = []
        if left_wrist:
            wrist_dists.append(math.sqrt((left_wrist.x - torso_center[0]) ** 2 +
                                          (left_wrist.y - torso_center[1]) ** 2))
        if right_wrist:
            wrist_dists.append(math.sqrt((right_wrist.x - torso_center[0]) ** 2 +
                                          (right_wrist.y - torso_center[1]) ** 2))
        if wrist_dists:
            arm_spread = sum(wrist_dists) / len(wrist_dists)

    # === Arm elevation (negative = raised above shoulder) ===
    left_arm_elevation = None
    right_arm_elevation = None
    if left_wrist and left_shoulder:
        left_arm_elevation = left_wrist.y - left_shoulder.y
    if right_wrist and right_shoulder:
        right_arm_elevation = right_wrist.y - right_shoulder.y

    # === Elbow forward position ===
    left_elbow_forward = None
    right_elbow_forward = None
    if left_elbow and left_shoulder:
        # Positive = elbow in front (assuming top-down or front view)
        left_elbow_forward = abs(left_elbow.y - left_shoulder.y)
    if right_elbow and right_shoulder:
        right_elbow_forward = abs(right_elbow.y - right_shoulder.y)

    # === Side asymmetry ===
    side_asymmetry = None
    if left_shoulder and right_shoulder and torso_center:
        # Measure how off-center the midpoint is
        shoulder_mid_x = (left_shoulder.x + right_shoulder.x) / 2
        if bounding_box:
            bbox_center_x = bounding_box[0] + bounding_box[2] / 2
            side_asymmetry = abs(shoulder_mid_x - bbox_center_x) / max(bounding_box[2], 1)

    # === Head tilt ===
    head_tilt = None
    if nose and left_shoulder and right_shoulder:
        shoulder_mid = ((left_shoulder.x + right_shoulder.x) / 2,
                        (left_shoulder.y + right_shoulder.y) / 2)
        dx = nose.x - shoulder_mid[0]
        dy = nose.y - shoulder_mid[1]
        if abs(dy) > 1:
            head_tilt = math.degrees(math.atan2(dx, -dy))  # 0 = upright

    # === Pose quality ===
    pose_quality = pose.torso_quality

    return PoseFeatures(
        torso_center=torso_center,
        torso_length=torso_length,
        head_to_hips_distance=head_to_hips_distance,
        body_compactness=body_compactness,
        aspect_ratio=aspect_ratio,
        bounding_box=bounding_box,
        left_elbow_angle=left_elbow_angle,
        right_elbow_angle=right_elbow_angle,
        left_wrist_to_head=left_wrist_to_head,
        right_wrist_to_head=right_wrist_to_head,
        arm_spread=arm_spread,
        left_arm_elevation=left_arm_elevation,
        right_arm_elevation=right_arm_elevation,
        left_elbow_forward=left_elbow_forward,
        right_elbow_forward=right_elbow_forward,
        side_asymmetry=side_asymmetry,
        head_tilt=head_tilt,
        pose_quality=pose_quality,
    )


def normalize_features(features: PoseFeatures, frame_size: Tuple[int, int]) -> PoseFeatures:
    """
    Normalize features relative to frame size for scale invariance.

    Args:
        features: Raw features
        frame_size: (width, height) of frame

    Returns:
        Normalized features
    """
    w, h = frame_size
    diag = math.sqrt(w ** 2 + h ** 2)

    def norm_dist(d: Optional[float]) -> Optional[float]:
        return d / diag if d is not None else None

    def norm_point(p: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        if p is None:
            return None
        return (p[0] / w, p[1] / h)

    return PoseFeatures(
        torso_center=norm_point(features.torso_center),
        torso_length=norm_dist(features.torso_length),
        head_to_hips_distance=norm_dist(features.head_to_hips_distance),
        body_compactness=features.body_compactness,  # Already ratio
        aspect_ratio=features.aspect_ratio,  # Already ratio
        bounding_box=features.bounding_box,  # Keep raw for now
        left_elbow_angle=features.left_elbow_angle,  # Already degrees
        right_elbow_angle=features.right_elbow_angle,
        left_wrist_to_head=norm_dist(features.left_wrist_to_head),
        right_wrist_to_head=norm_dist(features.right_wrist_to_head),
        arm_spread=norm_dist(features.arm_spread),
        left_arm_elevation=norm_dist(features.left_arm_elevation),
        right_arm_elevation=norm_dist(features.right_arm_elevation),
        left_elbow_forward=norm_dist(features.left_elbow_forward),
        right_elbow_forward=norm_dist(features.right_elbow_forward),
        side_asymmetry=features.side_asymmetry,  # Already ratio
        head_tilt=features.head_tilt,  # Already degrees
        pose_quality=features.pose_quality,
    )
