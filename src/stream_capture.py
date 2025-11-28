import cv2
import threading
import time
from typing import Optional
import numpy as np


class RTSPCapture:
    """Captures frames from an RTSP stream with reconnection handling."""

    def __init__(self, rtsp_url: str, reconnect_delay: float = 5.0):
        self.rtsp_url = rtsp_url
        self.reconnect_delay = reconnect_delay
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> bool:
        """Start capturing frames in a background thread."""
        if self._running:
            return True

        self._cap = cv2.VideoCapture(self.rtsp_url)
        if not self._cap.isOpened():
            print(f"Failed to open RTSP stream: {self.rtsp_url}")
            return False

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return True

    def _capture_loop(self):
        """Continuously capture frames from the stream."""
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                self._reconnect()
                continue

            ret, frame = self._cap.read()
            if not ret:
                print("Failed to read frame, attempting reconnect...")
                self._reconnect()
                continue

            with self._lock:
                self._frame = frame

    def _reconnect(self):
        """Attempt to reconnect to the RTSP stream."""
        if self._cap is not None:
            self._cap.release()

        print(f"Reconnecting in {self.reconnect_delay} seconds...")
        time.sleep(self.reconnect_delay)

        self._cap = cv2.VideoCapture(self.rtsp_url)
        if self._cap.isOpened():
            print("Reconnected successfully")
        else:
            print("Reconnection failed")

    def get_frame(self) -> Optional[np.ndarray]:
        """Get the most recent frame."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        """Stop capturing and release resources."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
