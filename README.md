# Baby Awake Detection

A Python application that monitors an RTSP video stream (designed for IR baby cameras) to determine whether a baby is awake or asleep using pose estimation and movement analysis.

## Features

- Real-time RTSP stream processing with automatic reconnection
- Optimized for IR/night vision cameras (CLAHE preprocessing)
- Movement-based detection (handles crib bounce filtering)
- Hysteresis-based state machine (prevents flickering)
- Visual overlay with state, detection rate, and transition progress
- Headless mode for running without display

## Requirements

- macOS (tested on Apple Silicon)
- Python 3.10-3.12 (MediaPipe does not support Python 3.13+)
- An RTSP camera stream (works best with top-down crib cameras)

## Setup

### 1. Install pyenv (if not already installed)

```bash
brew install pyenv
```

Add to your shell config:

```bash
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
source ~/.zshrc
```

### 2. Install Python 3.12

```bash
pyenv install 3.12.7
```

### 3. Set Python version for this project

```bash
cd /path/to/baby_awake_detection
pyenv local 3.12.7
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

### With video display

```bash
python main.py rtsp://192.168.1.100:554/stream
```

### Headless mode (no video window)

```bash
python main.py --no-video rtsp://192.168.1.100:554/stream
```

Press `q` to quit when the video window is displayed, or `Ctrl+C` to stop in headless mode.

## How It Works

### State Machine

The application uses a hysteresis-based state machine with three states:

```
EMPTY ──(2s detection)──> ASLEEP ──(5s movement)──> AWAKE
  ^                          ^                        |
  |                          |                        |
  +──(30s no detection)──────+────(60s still)─────────+
```

| State | Description |
|-------|-------------|
| **EMPTY** | No baby detected in crib |
| **ASLEEP** | Baby detected, lying still |
| **AWAKE** | Baby moving (crawling, rolling, etc.) |

### Transition Thresholds

| Transition | Requirement |
|------------|-------------|
| EMPTY → ASLEEP | Pose detected for 2+ seconds, no movement |
| EMPTY → AWAKE | Pose detected + movement |
| ASLEEP → AWAKE | Sustained movement for 5+ seconds |
| AWAKE → ASLEEP | Still for 60+ seconds |
| ANY → EMPTY | No detection for 30+ seconds |

### Key Design Decisions

- **Movement is the primary signal** - Eyes are rarely detectable with IR cameras
- **Relative movement detection** - Filters out camera shake and crib bounce
- **Detection accumulator** - Smooths over brief detection dropouts (common with IR)
- **Hysteresis prevents flickering** - State changes require sustained evidence

## Project Structure

```
baby_awake_detection/
├── main.py                     # Application entry point
├── requirements.txt            # Python dependencies
├── .python-version             # pyenv Python version
└── src/
    ├── stream_capture.py       # RTSP stream handling
    ├── pose_detector.py        # MediaPipe pose detection + IR preprocessing
    ├── detection_accumulator.py # Smoothing, dropout handling, movement calc
    ├── state_machine.py        # Hysteresis-based state transitions
    └── sleep_classifier.py     # Coordinator (ties everything together)
```

## Configuration

Thresholds can be adjusted in `main.py`:

```python
thresholds = TransitionThresholds(
    detection_to_present=2.0,    # Seconds to confirm baby present
    movement_to_awake=5.0,       # Seconds of movement to trigger awake
    still_to_asleep=60.0,        # Seconds still to trigger asleep
    no_detection_to_empty=30.0,  # Seconds without detection for empty
    movement_threshold=0.3,      # Movement score threshold (0-1)
)
```

## Troubleshooting

### MediaPipe installation fails

Ensure you're using Python 3.10-3.12:

```bash
python --version  # Should show 3.12.x
```

### Stream connection issues

- Verify the RTSP URL is correct
- Check network connectivity to the camera
- Some cameras require authentication in the URL: `rtsp://user:pass@ip:port/stream`

### Poor detection accuracy

- Ensure adequate lighting (IR illumination for night)
- Position camera for top-down view of crib
- Check detection rate in overlay (should be >50%)
- Adjust `min_detection_confidence` in `main.py` if needed (lower = more detections)

### Detection flickering

- If your IR camera has visible flicker, adjust camera settings (exposure, gain)
- The detection accumulator should smooth over brief dropouts automatically
