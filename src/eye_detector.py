import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from typing import Optional
from scipy.spatial import distance


@dataclass
class EyeResult:
    left_ear: float  # Eye Aspect Ratio for left eye
    right_ear: float  # Eye Aspect Ratio for right eye
    avg_ear: float  # Average EAR
    eyes_detected: bool
    eyes_open: bool  # Based on EAR threshold
    confidence: float


class EyeDetector:
    """Detects eyes and calculates Eye Aspect Ratio using MediaPipe Face Mesh."""

    # MediaPipe Face Mesh eye landmark indices
    # Left eye landmarks (from subject's perspective)
    LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    # Right eye landmarks
    RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

    def __init__(
        self,
        ear_threshold: float = 0.2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.ear_threshold = ear_threshold
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._ear_history: list[float] = []
        self._history_size = 5

    def detect(self, frame: np.ndarray) -> EyeResult:
        """Detect eyes and calculate EAR."""
        rgb_frame = frame[:, :, ::-1]  # BGR to RGB
        results = self._face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return EyeResult(
                left_ear=0.0,
                right_ear=0.0,
                avg_ear=0.0,
                eyes_detected=False,
                eyes_open=False,
                confidence=0.0,
            )

        face_landmarks = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]

        # Extract eye landmarks
        left_eye = self._get_eye_landmarks(face_landmarks, self.LEFT_EYE_INDICES, w, h)
        right_eye = self._get_eye_landmarks(face_landmarks, self.RIGHT_EYE_INDICES, w, h)

        # Calculate EAR for each eye
        left_ear = self._calculate_ear(left_eye)
        right_ear = self._calculate_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        # Smooth EAR with history
        self._ear_history.append(avg_ear)
        if len(self._ear_history) > self._history_size:
            self._ear_history.pop(0)
        smoothed_ear = np.mean(self._ear_history)

        # Determine if eyes are open
        eyes_open = smoothed_ear > self.ear_threshold

        # Calculate confidence based on landmark visibility
        confidence = self._calculate_confidence(face_landmarks, w, h)

        return EyeResult(
            left_ear=left_ear,
            right_ear=right_ear,
            avg_ear=smoothed_ear,
            eyes_detected=True,
            eyes_open=eyes_open,
            confidence=confidence,
        )

    def _get_eye_landmarks(
        self, face_landmarks, indices: list[int], width: int, height: int
    ) -> np.ndarray:
        """Extract eye landmark coordinates."""
        points = []
        for idx in indices:
            lm = face_landmarks.landmark[idx]
            points.append([lm.x * width, lm.y * height])
        return np.array(points)

    def _calculate_ear(self, eye_landmarks: np.ndarray) -> float:
        """
        Calculate Eye Aspect Ratio (EAR).

        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

        Where p1-p6 are the eye landmarks:
        p1, p4 are the horizontal corners
        p2, p3 are the upper lid points
        p5, p6 are the lower lid points
        """
        # Vertical distances
        v1 = distance.euclidean(eye_landmarks[1], eye_landmarks[5])
        v2 = distance.euclidean(eye_landmarks[2], eye_landmarks[4])

        # Horizontal distance
        h = distance.euclidean(eye_landmarks[0], eye_landmarks[3])

        if h == 0:
            return 0.0

        ear = (v1 + v2) / (2.0 * h)
        return ear

    def _calculate_confidence(self, face_landmarks, width: int, height: int) -> float:
        """Calculate detection confidence based on face position and visibility."""
        # Check if face is reasonably centered and sized
        nose = face_landmarks.landmark[1]  # Nose tip

        # Face should be within frame
        if nose.x < 0.1 or nose.x > 0.9 or nose.y < 0.1 or nose.y > 0.9:
            return 0.5

        # Higher confidence for centered faces
        center_dist = np.sqrt((nose.x - 0.5) ** 2 + (nose.y - 0.5) ** 2)
        confidence = max(0.5, 1.0 - center_dist)

        return confidence

    def close(self):
        """Release resources."""
        self._face_mesh.close()
