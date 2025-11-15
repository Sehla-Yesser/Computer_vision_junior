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

### 1. Person Detection

Detect persons in images or videos:

```bash
# Process an image
python src/detection/person_detector.py --input data/input/image.jpg --output data/output/result.jpg

# Process a video
python src/detection/person_detector.py --input data/input/video.mp4 --output data/output/result.mp4

# Use a different model and confidence threshold
python src/detection/person_detector.py --input data/input/image.jpg --output data/output/result.jpg --model yolov8m.pt --confidence 0.6
```

**Arguments:**
- `--input, -i`: Path to input image or video (required)
- `--output, -o`: Path to save output (optional)
- `--model, -m`: Path to YOLO model (default: yolov8n.pt)
- `--confidence, -c`: Confidence threshold (default: 0.5)

### 2. Facial Expression Recognition - YOLO Version

Detect persons and analyze facial expressions using YOLO:

```bash
# Process an image
python src/fer/fer_yolo.py --input data/input/image.jpg --output data/output/fer_result.jpg

# Process a video
python src/fer/fer_yolo.py --input data/input/video.mp4 --output data/output/fer_result.mp4

# Custom models
python src/fer/fer_yolo.py --input data/input/image.jpg --person-model yolov8m.pt --face-model yolov8n-face.pt --confidence 0.6
```

**Arguments:**
- `--input, -i`: Path to input image or video (required)
- `--output, -o`: Path to save output (optional)
- `--person-model`: Path to person detection YOLO model (default: yolov8n.pt)
- `--face-model`: Path to face detection YOLO model (default: yolov8n-face.pt)
- `--confidence, -c`: Confidence threshold (default: 0.5)

### 3. Facial Expression Recognition - MediaPipe Version

Detect persons and analyze facial expressions using MediaPipe:

```bash
# Process an image
python src/fer/fer_mediapipe.py --input data/input/image.jpg --output data/output/fer_mp_result.jpg

# Process a video
python src/fer/fer_mediapipe.py --input data/input/video.mp4 --output data/output/fer_mp_result.mp4

# With custom settings
python src/fer/fer_mediapipe.py --input data/input/image.jpg --person-model yolov8m.pt --confidence 0.6
```

**Arguments:**
- `--input, -i`: Path to input image or video (required)
- `--output, -o`: Path to save output (optional)
- `--person-model`: Path to person detection YOLO model (default: yolov8n.pt)
- `--confidence, -c`: Confidence threshold (default: 0.5)

## Python API Usage

You can also use the classes directly in your Python code:

```python
from src.detection.person_detector import PersonDetector
from src.fer.fer_yolo import FERYolo
from src.fer.fer_mediapipe import FERMediaPipe
import cv2

# Person Detection
detector = PersonDetector(model_path='yolov8n.pt', confidence=0.5)
image = cv2.imread('data/input/image.jpg')
detections = detector.detect_persons(image)
output_image = detector.detect_and_draw(image)

# FER with YOLO
fer_yolo = FERYolo(person_model='yolov8n.pt', confidence=0.5)
output_image, results = fer_yolo.process_image(image)

# FER with MediaPipe
fer_mp = FERMediaPipe(person_model='yolov8n.pt', confidence=0.5)
output_image, results = fer_mp.process_image(image)
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

- The FER implementations currently use placeholder emotion analysis. For production use, integrate specialized FER models or trained classifiers.
- MediaPipe provides more detailed facial landmarks which can be used for advanced expression analysis.
- YOLO approach is generally faster but may be less accurate for facial landmark detection.
- For best results with FER, ensure faces are clearly visible and well-lit.

## Future Enhancements

- Integration with specialized FER models (e.g., EmoNet, AffectNet)
- Real-time webcam support
- Multi-person tracking across frames
- Enhanced emotion classification
- Export results to JSON/CSV
- Web interface for easy usage

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

See LICENSE file for details.

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [MediaPipe](https://google.github.io/mediapipe/)
- OpenCV community
