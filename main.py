#!/usr/bin/env python3
"""Baby awake/asleep detection using RTSP stream and pose estimation."""

import argparse
import signal
import sys
import time

import cv2
import numpy as np

from src.stream_capture import RTSPCapture
from src.sleep_classifier import SleepClassifier, SleepState, TransitionThresholds


class BabyMonitor:
    """Main application for baby sleep state monitoring."""

    def __init__(self, rtsp_url: str, show_video: bool = True):
        self.rtsp_url = rtsp_url
        self.show_video = show_video

        self._capture = RTSPCapture(rtsp_url)

        # Configure thresholds (can be adjusted based on your setup)
        thresholds = TransitionThresholds(
            detection_to_present=2.0,    # 2s to confirm baby present
            movement_to_awake=5.0,       # 5s of movement to trigger awake
            still_to_asleep=60.0,        # 60s still to trigger asleep
            no_detection_to_empty=30.0,  # 30s no detection to trigger empty
            movement_threshold=0.3,      # Movement score threshold
        )

        self._classifier = SleepClassifier(
            min_detection_confidence=0.3,
            model_complexity=1,
            enable_ir_preprocessing=True,
            buffer_seconds=3.0,
            smoothing_factor=0.3,
            thresholds=thresholds,
        )

        self._running = False
        self._last_state = None

    def start(self):
        """Start monitoring."""
        if not self._capture.start():
            print("Failed to start stream capture")
            return False

        self._running = True
        print(f"Monitoring started on {self.rtsp_url}")
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

            # Process frame
            result = self._classifier.process_frame(frame)

            # Handle state changes
            if result.state != self._last_state:
                self._on_state_change(result.state, self._last_state, result.reason)
                self._last_state = result.state

            # Calculate FPS
            frame_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_start = time.time()

            # Display
            if self.show_video:
                display_frame = self._draw_overlay(frame, result, fps)
                cv2.imshow("Baby Monitor", display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

    def _on_state_change(self, new_state: SleepState, old_state: SleepState, reason: str):
        """Handle sleep state changes."""
        timestamp = time.strftime("%H:%M:%S")
        old_str = old_state.value if old_state else "startup"
        print(f"[{timestamp}] {old_str} -> {new_state.value}: {reason}")

    def _draw_overlay(self, frame: np.ndarray, result, fps: float) -> np.ndarray:
        """Draw detection overlay on frame."""
        display = frame.copy()
        h, w = display.shape[:2]

        # Draw skeleton if we have landmarks
        if result.landmarks is not None:
            detector = self._classifier.get_detector()
            display = detector.draw_skeleton(display, result.landmarks)

        # State indicator box (top-left)
        state_colors = {
            SleepState.AWAKE: (0, 255, 0),    # Green
            SleepState.ASLEEP: (255, 0, 0),   # Blue
            SleepState.EMPTY: (128, 128, 128), # Gray
        }
        color = state_colors.get(result.state, (128, 128, 128))

        # Draw state box
        cv2.rectangle(display, (10, 10), (200, 80), color, -1)
        cv2.rectangle(display, (10, 10), (200, 80), (255, 255, 255), 2)
        cv2.putText(display, result.state.value.upper(), (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Transition progress bar (if transitioning)
        if result.transition_progress and result.transition_target:
            bar_width = 180
            bar_height = 15
            bar_x = 10
            bar_y = 85
            progress = int(bar_width * result.transition_progress)

            # Background
            cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                         (50, 50, 50), -1)
            # Progress
            target_color = state_colors.get(result.transition_target, (200, 200, 200))
            cv2.rectangle(display, (bar_x, bar_y), (bar_x + progress, bar_y + bar_height),
                         target_color, -1)
            # Border
            cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                         (255, 255, 255), 1)
            # Label
            cv2.putText(display, f"-> {result.transition_target.value}",
                       (bar_x + bar_width + 5, bar_y + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Left panel - metrics
        y = 120
        metrics = [
            f"FPS: {fps:.1f}",
            f"Detection: {result.detection_rate*100:.0f}%",
            f"Movement: {result.movement_score:.2f}",
            f"Confidence: {result.confidence:.2f}",
            f"In state: {result.time_in_state:.1f}s",
        ]

        for text in metrics:
            cv2.putText(display, text, (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 22

        # Reason (below metrics)
        y += 10
        cv2.putText(display, f"Reason: {result.reason}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        return display

    def stop(self):
        """Stop monitoring and release resources."""
        self._running = False
        self._capture.stop()
        self._classifier.close()
        if self.show_video:
            cv2.destroyAllWindows()
        print("Monitoring stopped")


def main():
    parser = argparse.ArgumentParser(description="Baby awake/asleep detection from RTSP stream")
    parser.add_argument("rtsp_url", help="RTSP stream URL")
    parser.add_argument("--no-video", action="store_true", help="Disable video display")
    args = parser.parse_args()

    monitor = BabyMonitor(rtsp_url=args.rtsp_url, show_video=not args.no_video)

    def signal_handler(sig, frame):
        print("\nReceived signal, stopping...")
        monitor.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    monitor.start()


if __name__ == "__main__":
    main()
