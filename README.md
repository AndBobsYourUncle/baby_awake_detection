# Baby Awake Detection

A Python application that monitors an RTSP video stream (designed for IR baby cameras) to determine whether a baby is awake or asleep using pose estimation and movement analysis.

## Features

- **Pluggable pose estimation**: ViTPose (default) with YOLO pre-filter, or MediaPipe fallback
- **Pose classification**: Detects sitting, on-elbows, lying flat, etc. for immediate awake detection
- **Crib-relative motion detection**: Filters out camera shake and crib bounce
- **Arm/hand-aware**: Arms and hands weighted more heavily for awake detection
- **Hysteresis-based state machine**: Prevents flickering between states
- **Debug visualization**: Skeleton overlay with pose class, movement metrics
- **CSV logging**: Export frame-by-frame data for offline analysis
- **Configurable via YAML**: All thresholds and settings in one place

## Requirements

- macOS (tested on Apple Silicon) or Linux
- Python 3.10-3.12 (MediaPipe does not support Python 3.13+)
- An RTSP camera stream (works best with top-down crib cameras)

## Setup

### 1. Install Python 3.12 (if needed)

```bash
# Using pyenv
brew install pyenv
pyenv install 3.12.7
cd /path/to/baby_awake_detection
pyenv local 3.12.7
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Install ViTPose dependencies

For higher accuracy pose estimation:

```bash
pip install torch torchvision timm
```

## Usage

### Basic usage (ViTPose backend, default)

```bash
python main.py rtsp://192.168.1.100:554/stream
```

### With MediaPipe backend (fallback)

```bash
python main.py rtsp://192.168.1.100:554/stream --backend mediapipe
```

### Headless mode (no video window)

```bash
python main.py --no-video rtsp://192.168.1.100:554/stream
```

### With CSV logging

```bash
python main.py --log-csv rtsp://192.168.1.100:554/stream
```

### Using a config file

```bash
python main.py -c config.yaml rtsp://192.168.1.100:554/stream
```

### Save current config to file

```bash
python main.py --save-config config.yaml rtsp://192.168.1.100:554/stream
```

Press `q` to quit when the video window is displayed, or `Ctrl+C` to stop in headless mode.

## Architecture

### Pipeline

```
RTSP Stream → YOLO Filter → Pose Estimator → Feature Extraction → Pose Classifier
                  ↓                ↓                 ↓                    ↓
            (Person        (ViTPose or      (Geometric          (LYING_FLAT,
             detection)     MediaPipe)       features)           SITTING, etc.)
                                                    ↓
                                            Motion Analyzer → State Machine → Output
                                                  ↓                ↓
                                            (Crib-relative   (Hysteresis +
                                             torso/arm)       pose-aware)
```

### State Machine

```
EMPTY ──(2s good pose)──> ASLEEP ──(5s movement OR 1s awake pose)──> AWAKE
  ^                          ^                                         |
  |                          |                                         |
  +──(30s no detection)──────+────────(60s still + neutral pose)───────+
```

| State | Description |
|-------|-------------|
| **EMPTY** | No baby detected in crib |
| **ASLEEP** | Baby detected, lying still in neutral pose |
| **AWAKE** | Baby moving or in awake pose (sitting, on elbows, etc.) |

### Pose Classification

The system classifies the baby's static posture to detect awake states without requiring movement:

| Pose Class | Indicators | Awake? |
|------------|------------|--------|
| **LYING_FLAT** | Wide aspect ratio, low compactness | Neutral |
| **CURLED_TUCKED** | High compactness, wrists near head | Neutral |
| **SIDE_LYING** | Left-right asymmetry | Neutral |
| **ON_ELBOWS** | Bent elbows, elbows forward, arms elevated | **Awake** |
| **ON_ALL_FOURS** | Wide limb spread, compact body | **Awake** |
| **SITTING** | Upright aspect ratio, head close to hips | **Awake** |
| **STANDING** | Upright aspect ratio | **Awake** |

When an awake pose is detected for 1 second (configurable), the system immediately transitions to AWAKE state, even without significant movement.

### Motion Analysis

Movement is calculated by:

1. **Extracting keypoints** from pose estimator (torso + arms)
2. **Computing global motion** from torso keypoints (average movement)
3. **Subtracting global motion** from all keypoints (filters crib bounce)
4. **Separating torso vs arm movement**
5. **Weighting arms more heavily** (60% arms, 40% torso by default)
6. **Temporal smoothing** via EMA or median filter

This approach:
- Filters out camera shake and crib bounce
- Gives arm/hand movement priority (babies move arms a lot when awake)
- Provides stable readings even with detection dropouts

### Project Structure

```
baby_awake_detection/
├── main.py                     # Application entry point
├── requirements.txt            # Python dependencies
├── config.example.yaml         # Example configuration file
└── src/
    ├── stream_capture.py       # RTSP stream handling
    ├── pose_api.py             # Abstract pose estimator interface
    ├── mediapipe_estimator.py  # MediaPipe implementation
    ├── vitpose_estimator.py    # ViTPose + YOLO implementation
    ├── pose_features.py        # Geometric feature extraction
    ├── pose_classifier.py      # Rule-based pose classification
    ├── motion_analyzer.py      # Crib-relative motion detection
    ├── sleep_state_machine.py  # Hysteresis-based state transitions
    ├── debug_visualizer.py     # Overlay rendering & CSV logging
    └── config.py               # Configuration dataclasses
```

## Configuration

All settings can be configured via YAML file or command-line arguments.

### Example config.yaml

```yaml
pose:
  backend: vitpose  # or "mediapipe"
  vitpose_model: ViTPose-B
  vitpose_checkpoint: null  # Path to fine-tuned checkpoint
  enable_ir_preprocessing: true
  device: auto

motion:
  torso_weight: 0.4
  arm_weight: 0.6
  smoothing_window_seconds: 3.0
  movement_scale_pixels: 50.0
  buffer_seconds: 5.0

state:
  movement_low: 0.1       # Below this = "still"
  movement_high: 0.3      # Above this = "moving"
  movement_arms_high: 0.4 # Arm-specific awake threshold
  present_confirm_seconds: 2.0
  still_to_asleep_seconds: 60.0
  movement_to_awake_seconds: 5.0
  no_detection_to_empty_seconds: 30.0
  awake_pose_confirm_seconds: 1.0  # Awake pose for this long = awake

debug:
  show_overlay: true
  show_skeleton: true
  highlight_arms: true
  enable_csv_logging: false
  csv_log_path: motion_log.csv
```

### Key Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `motion.torso_weight` | 0.4 | Weight for torso in combined movement |
| `motion.arm_weight` | 0.6 | Weight for arms/hands (higher = more sensitive) |
| `state.movement_low` | 0.1 | Below this = "still" |
| `state.movement_high` | 0.3 | Above this = "moving" |
| `state.movement_arms_high` | 0.4 | Arm movement threshold for strong awake signal |
| `state.movement_to_awake_seconds` | 5.0 | Moving for this long triggers AWAKE |
| `state.still_to_asleep_seconds` | 60.0 | Still for this long triggers ASLEEP |
| `state.awake_pose_confirm_seconds` | 1.0 | Awake pose (sitting, etc.) triggers AWAKE |

## Debug Overlay

The video overlay shows:

- **State box** (top-left): Current state with color coding
- **Transition bar**: Progress toward next state
- **Metrics panel**:
  - FPS
  - Pose quality (% of torso keypoints visible)
  - Torso/arm keypoint counts
  - Pose classification (e.g., "Lying Flat", "Sitting [AWAKE]")
  - Movement scores (torso, arms, combined, smoothed)
  - Time in current state
- **Skeleton**: Torso in green, arms in yellow
- **Global motion arrow**: Shows detected crib bounce/camera shake

## CSV Logging

When enabled (`--log-csv`), exports per-frame data including:

- Timestamp
- Current state
- Pose quality and validation status
- Pose classification (class name, confidence, is_awake_pose)
- Movement scores (torso, arms, combined, smoothed)
- Transition progress
- Global motion vectors
- State reason

Useful for debugging misclassifications and tuning thresholds.

## Fine-tuning ViTPose

The architecture supports swapping in a fine-tuned ViTPose checkpoint:

```bash
python main.py --backend vitpose --checkpoint path/to/finetuned.pth rtsp://...
```

No code changes required - just point to your checkpoint file.

## Troubleshooting

### MediaPipe installation fails

Ensure you're using Python 3.10-3.12:

```bash
python --version  # Should show 3.12.x
```

### Stream connection issues

- Verify the RTSP URL is correct
- Check network connectivity to the camera
- Some cameras require authentication: `rtsp://user:pass@ip:port/stream`

### Poor detection accuracy

- Ensure adequate lighting (IR illumination for night)
- Position camera for top-down view of crib
- Check pose quality in overlay (should be >60%)
- Try ViTPose backend for better accuracy

### Movement detection not working

- Check movement metrics in overlay
- Lower `movement_high` threshold in config
- Ensure arms are visible to camera

### Detection flickering

- Increase `state.hysteresis_seconds` to prevent rapid transitions
- Check for IR camera flicker (adjust camera settings)
- Enable CSV logging to analyze the data
