"""
ViTPose-based pose estimator implementing the PoseEstimator interface.

This module provides a Vision Transformer-based pose estimation that is
more accurate than MediaPipe, especially for challenging views like
top-down crib cameras.

Requirements:
    pip install torch torchvision timm

For fine-tuning support, the checkpoint path can be swapped without
changing any other code.
"""

from typing import Optional, List, Tuple
import logging

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    transforms = None

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    timm = None

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    YOLO = None

from .pose_api import (
    PoseEstimator,
    PoseResult,
    Keypoint,
    KeypointName,
)
from .config import PoseConfig

logger = logging.getLogger(__name__)


# COCO keypoint ordering (standard for most pose models including ViTPose)
COCO_KEYPOINT_ORDER = [
    KeypointName.NOSE,
    KeypointName.LEFT_EYE,
    KeypointName.RIGHT_EYE,
    KeypointName.LEFT_EAR,
    KeypointName.RIGHT_EAR,
    KeypointName.LEFT_SHOULDER,
    KeypointName.RIGHT_SHOULDER,
    KeypointName.LEFT_ELBOW,
    KeypointName.RIGHT_ELBOW,
    KeypointName.LEFT_WRIST,
    KeypointName.RIGHT_WRIST,
    KeypointName.LEFT_HIP,
    KeypointName.RIGHT_HIP,
    KeypointName.LEFT_KNEE,
    KeypointName.RIGHT_KNEE,
    KeypointName.LEFT_ANKLE,
    KeypointName.RIGHT_ANKLE,
]


def get_keypoints_from_heatmaps(
    heatmaps: np.ndarray,
    original_size: Tuple[int, int],
) -> List[Tuple[float, float, float]]:
    """
    Extract keypoint coordinates and confidence from heatmaps.

    Args:
        heatmaps: Shape (num_keypoints, H, W)
        original_size: (width, height) of original image

    Returns:
        List of (x, y, confidence) tuples
    """
    num_keypoints, h, w = heatmaps.shape
    orig_w, orig_h = original_size

    keypoints = []
    for i in range(num_keypoints):
        heatmap = heatmaps[i]

        # Find maximum
        max_val = heatmap.max()
        if max_val > 0:
            max_idx = np.unravel_index(heatmap.argmax(), heatmap.shape)
            y_hm, x_hm = max_idx

            # Scale to original image coordinates
            x = x_hm / w * orig_w
            y = y_hm / h * orig_h

            # Confidence from heatmap value (apply sigmoid if needed)
            confidence = float(max_val)
        else:
            x, y, confidence = 0, 0, 0

        keypoints.append((x, y, confidence))

    return keypoints


class ViTPoseEstimator(PoseEstimator):
    """
    ViTPose-based implementation of PoseEstimator.

    Provides higher accuracy pose estimation using Vision Transformers,
    with support for custom fine-tuned checkpoints.
    """

    # Model variants and their timm backbone names
    MODEL_VARIANTS = {
        "ViTPose-S": "vit_small_patch16_224",
        "ViTPose-B": "vit_base_patch16_224",
        "ViTPose-L": "vit_large_patch16_224",
        "ViTPose-H": "vit_huge_patch14_224",
    }

    def __init__(self, config: PoseConfig):
        """
        Initialize ViTPose estimator.

        Args:
            config: Pose configuration
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for ViTPose. Install with: pip install torch torchvision")
        if not TIMM_AVAILABLE:
            raise ImportError("timm is required for ViTPose. Install with: pip install timm")

        self._config = config

        # Determine device
        if config.device == "auto":
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self._device = torch.device("mps")
            else:
                self._device = torch.device("cpu")
        else:
            self._device = torch.device(config.device)

        logger.info(f"ViTPose using device: {self._device}")

        # Get backbone name
        backbone_name = self.MODEL_VARIANTS.get(
            config.vitpose_model,
            "vit_base_patch16_224"
        )

        # Create model
        self._model = self._create_model(backbone_name)

        # Load custom checkpoint if provided
        if config.vitpose_checkpoint:
            logger.info(f"Loading checkpoint: {config.vitpose_checkpoint}")
            checkpoint = torch.load(config.vitpose_checkpoint, map_location=self._device)
            if 'state_dict' in checkpoint:
                self._model.load_state_dict(checkpoint['state_dict'])
            else:
                self._model.load_state_dict(checkpoint)

        self._model = self._model.to(self._device)
        self._model.eval()

        # Input transforms
        self._input_size = 224  # Standard ViT input size
        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self._input_size, self._input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # CLAHE for IR preprocessing
        if config.enable_ir_preprocessing:
            self._clahe = cv2.createCLAHE(
                clipLimit=config.clahe_clip_limit,
                tileGridSize=(config.clahe_grid_size, config.clahe_grid_size),
            )
        else:
            self._clahe = None

        # YOLO person detector (pre-filter to avoid false positives)
        self._person_detector = None
        self._person_conf_threshold = 0.3
        if YOLO_AVAILABLE:
            logger.info("Loading YOLOv8n for person detection...")
            self._person_detector = YOLO("yolov8n.pt")
            self._person_detector.to(self._device)
        else:
            logger.warning("YOLO not available - pose estimation may have false positives")

    def _create_model(self, backbone_name: str):
        """Create the ViTPose model with backbone and head."""

        class SimpleViTPoseHead(nn.Module):
            """Simple pose estimation head for ViT backbone."""

            def __init__(self, in_channels: int, num_keypoints: int = 17):
                super().__init__()
                self.num_keypoints = num_keypoints

                # Deconvolution layers to upsample features
                self.deconv_layers = nn.Sequential(
                    nn.ConvTranspose2d(in_channels, 256, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                )

                # Final layer to produce heatmaps
                self.final_layer = nn.Conv2d(256, num_keypoints, kernel_size=1)

            def forward(self, x):
                # x shape: (B, num_patches, embed_dim) from ViT
                B, N, C = x.shape

                # Reshape to spatial format (assuming square patch grid)
                H = W = int(N ** 0.5)
                x = x.transpose(1, 2).reshape(B, C, H, W)

                # Upsample
                x = self.deconv_layers(x)

                # Produce heatmaps
                heatmaps = self.final_layer(x)

                return heatmaps

        class ViTPoseModel(nn.Module):
            """Complete ViTPose model with backbone and head."""

            def __init__(self, backbone_name: str, num_keypoints: int = 17):
                super().__init__()

                # Load pretrained ViT backbone
                self.backbone = timm.create_model(
                    backbone_name,
                    pretrained=True,
                    num_classes=0,  # Remove classification head
                )

                # Get embedding dimension
                embed_dim = self.backbone.embed_dim

                # Create pose head
                self.head = SimpleViTPoseHead(embed_dim, num_keypoints)

            def forward(self, x):
                # Get features from backbone (without CLS token for some models)
                features = self.backbone.forward_features(x)

                # Remove CLS token if present
                if hasattr(self.backbone, 'num_prefix_tokens') and self.backbone.num_prefix_tokens > 0:
                    features = features[:, self.backbone.num_prefix_tokens:]

                # Get heatmaps from head
                heatmaps = self.head(features)

                return heatmaps

        return ViTPoseModel(backbone_name=backbone_name, num_keypoints=17)

    def estimate(self, frame_bgr: np.ndarray) -> Optional[PoseResult]:
        """
        Estimate pose from a BGR frame.

        Args:
            frame_bgr: OpenCV BGR image

        Returns:
            PoseResult if pose detected, None otherwise
        """
        h, w = frame_bgr.shape[:2]

        # First check if YOLO detects a person
        if self._person_detector is not None:
            if not self._detect_person(frame_bgr):
                return None

        # Preprocess for IR cameras
        if self._clahe is not None:
            frame_bgr = self._preprocess_ir(frame_bgr)

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Apply transforms
        input_tensor = self._transform(frame_rgb).unsqueeze(0).to(self._device)

        # Run inference
        with torch.no_grad():
            heatmaps = self._model(input_tensor)

        # Convert heatmaps to numpy
        heatmaps_np = heatmaps[0].cpu().numpy()

        # Apply sigmoid to get confidence values
        heatmaps_np = 1 / (1 + np.exp(-heatmaps_np))

        # Extract keypoints
        kp_coords = get_keypoints_from_heatmaps(heatmaps_np, (w, h))

        # Convert to our format
        keypoints = {}
        total_confidence = 0.0
        num_valid = 0

        for i, kp_name in enumerate(COCO_KEYPOINT_ORDER):
            x, y, conf = kp_coords[i]
            keypoints[kp_name] = Keypoint(
                name=kp_name,
                x=x,
                y=y,
                confidence=conf,
            )
            if conf > 0.5:  # Stricter threshold
                total_confidence += conf
                num_valid += 1

        # Calculate overall score
        score = total_confidence / max(num_valid, 1) if num_valid > 0 else 0.0

        # Validate detection quality to filter out garbage
        if not self._is_valid_detection(keypoints, score, num_valid):
            return None

        return PoseResult(
            keypoints=keypoints,
            score=score,
        )

    def _is_valid_detection(self, keypoints: dict, score: float, num_valid: int) -> bool:
        """
        Validate that detection is a real person, not random noise.

        Args:
            keypoints: Dict of detected keypoints
            score: Overall pose score
            num_valid: Number of high-confidence keypoints

        Returns:
            True if detection appears to be a real person
        """
        # Need minimum number of confident keypoints
        if num_valid < 5:
            return False

        # Need minimum overall confidence
        if score < 0.4:
            return False

        # Check for key body parts (at least some torso keypoints)
        torso_names = [
            KeypointName.NOSE,
            KeypointName.LEFT_SHOULDER,
            KeypointName.RIGHT_SHOULDER,
            KeypointName.LEFT_HIP,
            KeypointName.RIGHT_HIP,
        ]
        torso_visible = sum(
            1 for name in torso_names
            if name in keypoints and keypoints[name].confidence > 0.4
        )
        if torso_visible < 2:
            return False

        # Check spatial coherence - shoulders should be reasonably close together
        left_shoulder = keypoints.get(KeypointName.LEFT_SHOULDER)
        right_shoulder = keypoints.get(KeypointName.RIGHT_SHOULDER)
        if left_shoulder and right_shoulder:
            if left_shoulder.confidence > 0.3 and right_shoulder.confidence > 0.3:
                shoulder_dist = np.sqrt(
                    (left_shoulder.x - right_shoulder.x) ** 2 +
                    (left_shoulder.y - right_shoulder.y) ** 2
                )
                # Shoulders shouldn't be too far apart (more than half the frame)
                # or too close (less than 10 pixels)
                frame_diag = np.sqrt(self._input_size ** 2 * 2)
                if shoulder_dist > frame_diag * 0.7 or shoulder_dist < 10:
                    return False

        return True

    def _detect_person(self, frame_bgr: np.ndarray) -> bool:
        """
        Use YOLO to check if a person is present in the frame.

        Args:
            frame_bgr: OpenCV BGR image

        Returns:
            True if person detected with sufficient confidence
        """
        # Run YOLO inference
        results = self._person_detector(frame_bgr, verbose=False)

        # Check for person detections (class 0 in COCO)
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    # Class 0 is "person" in COCO
                    if cls == 0 and conf >= self._person_conf_threshold:
                        return True

        return False

    def _preprocess_ir(self, frame: np.ndarray) -> np.ndarray:
        """Apply preprocessing to improve detection on IR/grayscale cameras."""
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self._clahe.apply(l)
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return enhanced

    def close(self):
        """Release resources."""
        if hasattr(self, '_model'):
            del self._model
        if hasattr(self, '_person_detector') and self._person_detector is not None:
            del self._person_detector
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()


def create_pose_estimator(config: PoseConfig) -> PoseEstimator:
    """
    Factory function to create appropriate pose estimator.

    Args:
        config: Pose configuration

    Returns:
        PoseEstimator instance
    """
    if config.backend == "vitpose":
        if not TORCH_AVAILABLE:
            raise ImportError(
                "ViTPose requires PyTorch. Install with:\n"
                "  pip install torch torchvision timm"
            )
        if not TIMM_AVAILABLE:
            raise ImportError(
                "ViTPose requires timm. Install with:\n"
                "  pip install timm"
            )
        return ViTPoseEstimator(config)

    if config.backend == "mediapipe":
        from .mediapipe_estimator import MediaPipeEstimator
        return MediaPipeEstimator(config)

    raise ValueError(f"Unknown pose backend: {config.backend}")
