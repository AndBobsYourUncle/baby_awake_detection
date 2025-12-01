"""
Configuration system for baby awake detection.

All configurable parameters are defined here. Configuration can be loaded
from a YAML file or constructed programmatically.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import socket
import yaml


@dataclass
class PoseConfig:
    """Configuration for pose estimation."""
    # Model selection - ViTPose is primary, MediaPipe only if explicitly chosen
    backend: str = "vitpose"

    # ViTPose settings
    vitpose_model: str = "ViTPose-B"  # Model variant: ViTPose-S, ViTPose-B, ViTPose-L, ViTPose-H
    vitpose_checkpoint: Optional[str] = None  # Path to checkpoint (None = download default)
    vitpose_config: Optional[str] = None  # Path to config file (None = use default)

    # MediaPipe fallback settings
    mediapipe_model_complexity: int = 1  # 0=lite, 1=full, 2=heavy
    mediapipe_min_detection_confidence: float = 0.3
    mediapipe_min_tracking_confidence: float = 0.3

    # Preprocessing
    enable_ir_preprocessing: bool = True  # CLAHE for IR cameras
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8

    # Device
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"


@dataclass
class MotionConfig:
    """Configuration for motion analysis."""
    # Pose quality thresholds
    min_torso_keypoints: int = 3  # Minimum torso keypoints to consider pose valid
    min_keypoint_confidence: float = 0.3  # Minimum confidence per keypoint

    # Movement weighting
    torso_weight: float = 0.4  # Weight for torso movement in overall score
    arm_weight: float = 0.6  # Weight for arm/hand movement (higher = more sensitive to arms)

    # Temporal smoothing
    smoothing_window_seconds: float = 3.0  # Window for movement smoothing
    smoothing_type: str = "ema"  # "ema" (exponential moving average) or "median"
    ema_alpha: float = 0.3  # EMA smoothing factor (lower = more smoothing)

    # Movement normalization
    movement_scale_pixels: float = 50.0  # Pixel movement that maps to score of 1.0

    # Buffer settings
    buffer_seconds: float = 5.0  # How long to hold pose estimate during dropouts


@dataclass
class StateConfig:
    """Configuration for state machine transitions."""
    # Movement thresholds
    movement_low: float = 0.1  # Below this = "still" (for sleep detection)
    movement_high: float = 0.3  # Above this = "moving" (for awake detection)
    movement_arms_high: float = 0.4  # Arm-specific threshold for strong awake signal

    # Time thresholds (in seconds)
    present_confirm_seconds: float = 2.0  # Good pose for this long = confirmed present
    still_to_asleep_seconds: float = 60.0  # Still for this long = asleep
    movement_to_awake_seconds: float = 5.0  # Moving for this long = awake
    no_detection_to_empty_seconds: float = 30.0  # No pose for this long = empty
    awake_pose_confirm_seconds: float = 1.0  # Awake pose (sitting, etc.) for this long = awake

    # Hysteresis buffer
    hysteresis_seconds: float = 2.0  # Minimum time between state changes


@dataclass
class DebugConfig:
    """Configuration for debugging and visualization."""
    # Overlay
    show_overlay: bool = True
    show_skeleton: bool = True
    show_keypoint_names: bool = False
    show_confidence: bool = True
    highlight_arms: bool = True  # Draw arms in different color

    # Colors (BGR)
    torso_color: tuple = (0, 255, 0)  # Green
    arm_color: tuple = (0, 255, 255)  # Yellow
    low_confidence_color: tuple = (0, 0, 255)  # Red

    # Logging
    enable_csv_logging: bool = False
    csv_log_path: str = "motion_log.csv"
    log_interval_seconds: float = 0.1  # How often to log (0 = every frame)


def _default_client_id() -> str:
    """Generate default MQTT client ID based on hostname."""
    hostname = socket.gethostname().replace('.', '_').replace('-', '_')[:20]
    return f"baby_awake_{hostname}"


@dataclass
class MqttConfig:
    """Configuration for MQTT integration with Home Assistant."""
    # Enable/disable MQTT
    enabled: bool = False

    # Broker connection
    host: str = "localhost"
    port: int = 1883
    username: Optional[str] = None
    password: Optional[str] = None
    tls: bool = False

    # Client settings
    client_id: str = field(default_factory=_default_client_id)

    # Home Assistant discovery
    discovery_prefix: str = "homeassistant"
    base_topic: str = "babycam/crib"

    # Device identification
    device_id: str = "baby_crib_monitor"
    device_name: str = "Baby Crib Monitor"

    # Publishing settings
    publish_interval_seconds: float = 1.0  # How often to publish state (0 = every frame)
    qos: int = 1  # MQTT QoS level (0, 1, or 2)


@dataclass
class Config:
    """Main configuration container."""
    pose: PoseConfig = field(default_factory=PoseConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    state: StateConfig = field(default_factory=StateConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    mqtt: MqttConfig = field(default_factory=MqttConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from a YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        config = cls()

        if 'pose' in data:
            for key, value in data['pose'].items():
                if hasattr(config.pose, key):
                    setattr(config.pose, key, value)

        if 'motion' in data:
            for key, value in data['motion'].items():
                if hasattr(config.motion, key):
                    setattr(config.motion, key, value)

        if 'state' in data:
            for key, value in data['state'].items():
                if hasattr(config.state, key):
                    setattr(config.state, key, value)

        if 'debug' in data:
            for key, value in data['debug'].items():
                if hasattr(config.debug, key):
                    setattr(config.debug, key, value)

        if 'mqtt' in data:
            for key, value in data['mqtt'].items():
                if hasattr(config.mqtt, key):
                    setattr(config.mqtt, key, value)

        return config

    def to_yaml(self, path: str):
        """Save configuration to a YAML file."""
        data = {
            'pose': {
                'backend': self.pose.backend,
                'vitpose_model': self.pose.vitpose_model,
                'vitpose_checkpoint': self.pose.vitpose_checkpoint,
                'vitpose_config': self.pose.vitpose_config,
                'mediapipe_model_complexity': self.pose.mediapipe_model_complexity,
                'mediapipe_min_detection_confidence': self.pose.mediapipe_min_detection_confidence,
                'enable_ir_preprocessing': self.pose.enable_ir_preprocessing,
                'device': self.pose.device,
            },
            'motion': {
                'min_torso_keypoints': self.motion.min_torso_keypoints,
                'min_keypoint_confidence': self.motion.min_keypoint_confidence,
                'torso_weight': self.motion.torso_weight,
                'arm_weight': self.motion.arm_weight,
                'smoothing_window_seconds': self.motion.smoothing_window_seconds,
                'ema_alpha': self.motion.ema_alpha,
                'movement_scale_pixels': self.motion.movement_scale_pixels,
                'buffer_seconds': self.motion.buffer_seconds,
            },
            'state': {
                'movement_low': self.state.movement_low,
                'movement_high': self.state.movement_high,
                'movement_arms_high': self.state.movement_arms_high,
                'present_confirm_seconds': self.state.present_confirm_seconds,
                'still_to_asleep_seconds': self.state.still_to_asleep_seconds,
                'movement_to_awake_seconds': self.state.movement_to_awake_seconds,
                'no_detection_to_empty_seconds': self.state.no_detection_to_empty_seconds,
                'awake_pose_confirm_seconds': self.state.awake_pose_confirm_seconds,
                'hysteresis_seconds': self.state.hysteresis_seconds,
            },
            'debug': {
                'show_overlay': self.debug.show_overlay,
                'show_skeleton': self.debug.show_skeleton,
                'highlight_arms': self.debug.highlight_arms,
                'enable_csv_logging': self.debug.enable_csv_logging,
                'csv_log_path': self.debug.csv_log_path,
            },
            'mqtt': {
                'enabled': self.mqtt.enabled,
                'host': self.mqtt.host,
                'port': self.mqtt.port,
                'username': self.mqtt.username,
                'password': self.mqtt.password,
                'tls': self.mqtt.tls,
                'client_id': self.mqtt.client_id,
                'discovery_prefix': self.mqtt.discovery_prefix,
                'base_topic': self.mqtt.base_topic,
                'device_id': self.mqtt.device_id,
                'device_name': self.mqtt.device_name,
                'publish_interval_seconds': self.mqtt.publish_interval_seconds,
                'qos': self.mqtt.qos,
            },
        }

        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
