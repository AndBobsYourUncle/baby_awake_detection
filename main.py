#!/usr/bin/env python3
"""
Baby awake/asleep detection using RTSP stream and pose estimation.

New architecture:
- Pluggable pose estimator (ViTPose or MediaPipe fallback)
- Crib-relative motion detection with arm/torso separation
- Hysteresis-based state machine with arm-aware logic
- Optional MQTT integration with Home Assistant
- Configurable via YAML or command-line
"""

import argparse
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import cv2

from src.stream_capture import RTSPCapture
from src.config import Config, PoseConfig, MotionConfig, StateConfig, DebugConfig
from src.vitpose_estimator import create_pose_estimator
from src.motion_analyzer import MotionAnalyzer
from src.sleep_state_machine import SleepStateMachine, SleepState, StateThresholds
from src.debug_visualizer import DebugVisualizer
from src.state_snapshot import StateSnapshot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class BabyMonitor:
    """Main application for baby sleep state monitoring."""

    def __init__(
        self,
        rtsp_url: str,
        config: Config,
        show_video: bool = True,
    ):
        """
        Initialize baby monitor.

        Args:
            rtsp_url: RTSP stream URL
            config: Application configuration
            show_video: Whether to display video window
        """
        self.rtsp_url = rtsp_url
        self.show_video = show_video
        self._config = config

        # Stream capture
        self._capture = RTSPCapture(rtsp_url)

        # Pose estimator (ViTPose or MediaPipe fallback)
        print(f"Initializing pose estimator ({config.pose.backend})...")
        self._estimator = create_pose_estimator(config.pose)

        # Motion analyzer
        self._motion = MotionAnalyzer(config.motion)

        # State machine
        state_thresholds = StateThresholds(
            movement_low=config.state.movement_low,
            movement_high=config.state.movement_high,
            movement_arms_high=config.state.movement_arms_high,
            present_confirm_seconds=config.state.present_confirm_seconds,
            still_to_asleep_seconds=config.state.still_to_asleep_seconds,
            movement_to_awake_seconds=config.state.movement_to_awake_seconds,
            no_detection_to_empty_seconds=config.state.no_detection_to_empty_seconds,
            awake_pose_confirm_seconds=config.state.awake_pose_confirm_seconds,
            hysteresis_seconds=config.state.hysteresis_seconds,
        )
        self._state_machine = SleepStateMachine(thresholds=state_thresholds)

        # Debug visualizer
        self._visualizer = DebugVisualizer(config.debug, self._estimator)

        # MQTT publisher (optional)
        self._mqtt: Optional["MqttPublisher"] = None
        self._last_mqtt_publish = 0.0
        if config.mqtt.enabled:
            self._init_mqtt()

        self._running = False
        self._last_state = None

    def _init_mqtt(self):
        """Initialize MQTT publisher if enabled."""
        try:
            from src.mqtt_client import MqttPublisher
            self._mqtt = MqttPublisher(self._config.mqtt)
            if self._mqtt.connect():
                print(f"MQTT enabled: {self._config.mqtt.host}:{self._config.mqtt.port}")
            else:
                print("Warning: MQTT connection failed, continuing without MQTT")
                self._mqtt = None
        except ImportError as e:
            print(f"Warning: MQTT not available ({e}), continuing without MQTT")

    def start(self):
        """Start monitoring."""
        if not self._capture.start():
            print("Failed to start stream capture")
            return False

        self._running = True
        print(f"Monitoring started on {self.rtsp_url}")
        print(f"Pose backend: {self._config.pose.backend}")
        print("Press 'q' to quit")
        print("-" * 50)

        try:
            self._run_loop()
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.stop()

        return True

    def _run_loop(self):
        """Main processing loop."""
        frame_count = 0
        fps_start = time.time()
        fps = 0.0

        while self._running:
            frame = self._capture.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            timestamp = time.time()

            # 1. Detect pose
            pose = self._estimator.estimate(frame)

            # 2. Analyze motion
            motion = self._motion.update(pose, timestamp)

            # 3. Update state machine
            arm_movement = self._motion.get_smoothed_arm_movement()
            state_info = self._state_machine.update(
                baby_present=motion.baby_present,
                smoothed_movement=motion.smoothed_movement,
                arm_movement=arm_movement,
                seconds_since_detection=motion.seconds_since_detection,
                is_awake_pose=motion.is_awake_pose,
                timestamp=timestamp,
            )

            # Handle state changes
            if state_info.state != self._last_state:
                self._on_state_change(state_info.state, self._last_state, state_info.reason)
                self._last_state = state_info.state

            # Log to CSV
            self._visualizer.log_frame(motion, state_info, timestamp)

            # Publish to MQTT (rate limited)
            self._publish_mqtt(motion, state_info, timestamp)

            # Calculate FPS
            frame_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_start = time.time()

            # Display
            if self.show_video:
                display_frame = self._visualizer.draw_overlay(
                    frame, pose, motion, state_info, fps
                )
                cv2.imshow("Baby Monitor", display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

    def _publish_mqtt(self, motion, state_info, timestamp: float):
        """Publish state to MQTT if enabled and rate limit allows."""
        if not self._mqtt:
            return

        # Rate limit publishing
        interval = self._config.mqtt.publish_interval_seconds
        if interval > 0 and (timestamp - self._last_mqtt_publish) < interval:
            return

        self._last_mqtt_publish = timestamp

        # Create snapshot and publish
        try:
            snapshot = StateSnapshot.from_state(motion, state_info, timestamp)
            self._mqtt.publish_state(snapshot)
        except Exception as e:
            # Don't let MQTT errors crash the main loop
            logging.getLogger(__name__).error(f"MQTT publish error: {e}")

    def _on_state_change(
        self,
        new_state: SleepState,
        old_state: SleepState,
        reason: str,
    ):
        """Handle sleep state changes."""
        timestamp = time.strftime("%H:%M:%S")
        old_str = old_state.value if old_state else "startup"
        print(f"[{timestamp}] {old_str} -> {new_state.value}: {reason}")

    def stop(self):
        """Stop monitoring and release resources."""
        self._running = False
        self._capture.stop()
        self._estimator.close()
        self._visualizer.close()
        if self._mqtt:
            self._mqtt.disconnect()
        if self.show_video:
            cv2.destroyAllWindows()
        print("Monitoring stopped")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Baby awake/asleep detection from RTSP stream"
    )
    parser.add_argument("rtsp_url", help="RTSP stream URL")
    parser.add_argument(
        "--config",
        "-c",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Disable video display (headless mode)",
    )
    parser.add_argument(
        "--backend",
        choices=["vitpose", "mediapipe"],
        default="vitpose",
        help="Pose estimation backend (default: vitpose)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device for ViTPose (default: auto)",
    )
    parser.add_argument(
        "--vitpose-model",
        choices=["ViTPose-S", "ViTPose-B", "ViTPose-L", "ViTPose-H"],
        default="ViTPose-B",
        help="ViTPose model variant (default: ViTPose-B)",
    )
    parser.add_argument(
        "--checkpoint",
        help="Path to custom ViTPose checkpoint",
    )
    parser.add_argument(
        "--log-csv",
        action="store_true",
        help="Enable CSV logging",
    )
    parser.add_argument(
        "--save-config",
        help="Save current config to YAML file and exit",
    )

    # MQTT options
    parser.add_argument(
        "--mqtt",
        action="store_true",
        help="Enable MQTT publishing to Home Assistant",
    )
    parser.add_argument(
        "--mqtt-host",
        help="MQTT broker hostname (default: localhost)",
    )
    parser.add_argument(
        "--mqtt-port",
        type=int,
        help="MQTT broker port (default: 1883)",
    )
    parser.add_argument(
        "--mqtt-user",
        help="MQTT username",
    )
    parser.add_argument(
        "--mqtt-password",
        help="MQTT password",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Build configuration
    if args.config and Path(args.config).exists():
        config = Config.from_yaml(args.config)
    else:
        config = Config()

    # Override with command-line args
    config.pose.backend = args.backend
    config.pose.device = args.device
    config.pose.vitpose_model = args.vitpose_model
    if args.checkpoint:
        config.pose.vitpose_checkpoint = args.checkpoint
    if args.log_csv:
        config.debug.enable_csv_logging = True

    # MQTT overrides
    if args.mqtt:
        config.mqtt.enabled = True
    if args.mqtt_host:
        config.mqtt.host = args.mqtt_host
    if args.mqtt_port:
        config.mqtt.port = args.mqtt_port
    if args.mqtt_user:
        config.mqtt.username = args.mqtt_user
    if args.mqtt_password:
        config.mqtt.password = args.mqtt_password

    # Save config if requested
    if args.save_config:
        config.to_yaml(args.save_config)
        print(f"Configuration saved to {args.save_config}")
        return

    # Create and start monitor
    monitor = BabyMonitor(
        rtsp_url=args.rtsp_url,
        config=config,
        show_video=not args.no_video,
    )

    def signal_handler(sig, frame):
        print("\nReceived signal, stopping...")
        monitor.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    monitor.start()


if __name__ == "__main__":
    main()
