# Baby Awake Detection

A Python application that monitors an RTSP video stream to determine whether a baby is awake or asleep using pose estimation and Eye Aspect Ratio (EAR) analysis.

## Features

- Real-time RTSP stream processing with automatic reconnection
- Body pose detection (lying down, sitting, standing, moving)
- Eye state detection using Eye Aspect Ratio (EAR)
- Sleep state classification: AWAKE, DROWSY, ASLEEP, UNKNOWN
- Visual overlay showing detection metrics
- Headless mode for running without display

## Requirements

- macOS (tested on Apple Silicon)
- Python 3.10-3.12 (MediaPipe does not support Python 3.13+)
- An RTSP camera stream

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

The application combines three signals to classify sleep state:

| Signal | Weight | Description |
|--------|--------|-------------|
| Eye Aspect Ratio (EAR) | 40% | Ratio of eye height to width - lower values indicate closed eyes |
| Body Posture | 30% | Lying down suggests sleep, sitting/standing suggests awake |
| Movement | 30% | More movement indicates wakefulness |

### Sleep States

- **AWAKE**: Eyes open, upright posture, or significant movement
- **DROWSY**: Transitional state between awake and asleep
- **ASLEEP**: Eyes closed, lying down, minimal movement
- **UNKNOWN**: Insufficient detection confidence

State changes require 3 seconds of persistence to prevent flickering.

## Project Structure

```
baby_awake_detection/
├── main.py                 # Application entry point
├── requirements.txt        # Python dependencies
├── .python-version         # pyenv Python version
└── src/
    ├── stream_capture.py   # RTSP stream handling
    ├── pose_detector.py    # MediaPipe pose detection
    ├── eye_detector.py     # Face mesh + EAR calculation
    └── sleep_classifier.py # Sleep state classification
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

- Ensure adequate lighting
- Position camera to capture full body and face
- Adjust `ear_threshold` in `eye_detector.py` if eye detection is unreliable
