# Person Detector

This repository includes code for detecting humans (presence + behavior/sentiment analysis) using YOLO and facial expression recognition using both YOLO and MediaPipe approaches.

## Features

- **Person Detection**: Detect persons in images and videos using YOLOv8
- **Facial Expression Recognition (FER)**: Two approaches for analyzing facial expressions:
  - YOLO-based FER: Uses YOLO models for both person and face detection
  - MediaPipe-based FER: Uses YOLO for person detection and MediaPipe for face detection and landmark analysis

## Project Structure

```
Person_detector/
├── src/
│   ├── detection/
│   │   ├── __init__.py
│   │   └── person_detector.py      # Person detection using YOLO
│   └── fer/
│       ├── __init__.py
│       ├── fer_yolo.py             # FER using YOLO
│       └── fer_mediapipe.py        # FER using MediaPipe
├── data/
│   ├── input/                      # Place input images/videos here
│   └── output/                     # Processed outputs saved here
├── models/                         # YOLO model weights stored here
├── tests/                          # Unit tests
├── requirements.txt                # Python dependencies
├── .gitignore
└── README.md
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Sehla-Yesser/Person_detector.git
cd Person_detector
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download YOLO models (optional - will auto-download on first run):
```bash
# The models will be automatically downloaded when you first run the scripts
# Default model: yolov8n.pt (nano - fastest, least accurate)
# Available models: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
```

## Usage

### Basic Webcam Detection

Start real-time person detection with your webcam:

```bash
# Start webcam detection (press 'q' to quit)
python src/detection/person_detector.py

# Or explicitly specify webcam
python src/detection/person_detector.py --webcam

# Save webcam recording with detections
python src/detection/person_detector.py --webcam --output data/output/recording.mp4

# Use a different model and confidence threshold
python src/detection/person_detector.py --webcam --model yolov8m.pt --confidence 0.6

# Use a different camera (if you have multiple cameras)
python src/detection/person_detector.py --webcam --camera 1
```

**Arguments:**
- `--webcam, -w`: Use webcam for real-time detection
- `--output, -o`: Path to save output video (optional)
- `--model, -m`: Path to YOLO model (default: yolov8n.pt)
- `--confidence, -c`: Confidence threshold (default: 0.5)
- `--camera`: Camera device ID (default: 0)

### Run Examples

Try the interactive examples:

```bash
cd examples
python example_person_detection.py
```

This will show you a menu with different webcam detection options:
1. Basic Webcam Detection
2. Webcam Detection with Recording
3. Custom Webcam Detection with Statistics

## Python API Usage

You can also use the PersonDetector class directly in your Python code:

```python
from src.detection.person_detector import PersonDetector
import cv2

# Initialize detector
detector = PersonDetector(model_path='yolov8n.pt', confidence=0.5)

# Process webcam
detector.process_webcam(camera_id=0, save_output=False)

# Or process individual frames
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if ret:
    detections = detector.detect_persons(frame)
    output_frame = detector.detect_and_draw(frame)
    cv2.imshow('Detection', output_frame)
cap.release()
```

## Model Information

### YOLO Models
- **yolov8n.pt**: Nano - Fastest, least accurate
- **yolov8s.pt**: Small - Good balance
- **yolov8m.pt**: Medium - Better accuracy
- **yolov8l.pt**: Large - High accuracy
- **yolov8x.pt**: Extra Large - Best accuracy, slowest

Models are automatically downloaded on first use.

## Notes

- Works with any USB webcam or built-in laptop camera
- Press 'q' in the video window to quit
- Models are automatically downloaded on first use
- GPU acceleration is used automatically if CUDA is available
- For best results, ensure good lighting conditions

## Controls

- **'q'**: Quit the application
- **ESC**: Alternative quit key

## Future Enhancements

- Multi-person tracking across frames
- Person re-identification
- Activity recognition
- Export detection data to JSON/CSV
- Web interface for remote monitoring
- Support for IP cameras

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

See LICENSE file for details.

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [MediaPipe](https://google.github.io/mediapipe/)
- OpenCV community
