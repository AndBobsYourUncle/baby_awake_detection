#!/usr/bin/env python3
"""
Baby awake/asleep detection using RTSP stream and pose estimation.

New architecture:
- Pluggable pose estimator (ViTPose or MediaPipe fallback)
- Crib-relative motion detection with arm/torso separation
- Hysteresis-based state machine with arm-aware logic
- Configurable via YAML or command-line
"""

import argparse
import signal
import sys
import time
from pathlib import Path

import cv2

from src.stream_capture import RTSPCapture
from src.config import Config, PoseConfig, MotionConfig, StateConfig, DebugConfig
from src.vitpose_estimator import create_pose_estimator
from src.motion_analyzer import MotionAnalyzer
from src.sleep_state_machine import SleepStateMachine, SleepState, StateThresholds
from src.debug_visualizer import DebugVisualizer


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

        self._running = False
        self._last_state = None

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
