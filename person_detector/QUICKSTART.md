# Quick Start Guide

This guide will help you get started with Person Detector in just a few minutes.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster processing

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/Sehla-Yesser/Person_detector.git
cd Person_detector
```

### 2. Create Virtual Environment (Recommended)

**On Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:
- ultralytics (YOLOv8)
- opencv-python
- mediapipe
- torch and torchvision
- numpy, pillow, and other utilities

**Note:** The first time you run a script, YOLO will automatically download the model weights (~6MB for yolov8n.pt).

## First Run - Quick Test

### Test Person Detection

1. Place an image with people in `data/input/` directory
2. Run the person detector:

```bash
python src/detection/person_detector.py --input data/input/your_image.jpg --output data/output/result.jpg
```

3. Check the output in `data/output/result.jpg`

### Test Facial Expression Recognition (MediaPipe)

```bash
python src/fer/fer_mediapipe.py --input data/input/your_image.jpg --output data/output/fer_result.jpg
```

### Run the Demo

```bash
python demo.py
```

The demo will process any images in `data/input/` and show you all the capabilities.

## Basic Usage Examples

### Person Detection in Image

```bash
python src/detection/person_detector.py \
    --input data/input/image.jpg \
    --output data/output/detected.jpg \
    --confidence 0.6
```

### Person Detection in Video

```bash
python src/detection/person_detector.py \
    --input data/input/video.mp4 \
    --output data/output/detected_video.mp4
```

### FER with YOLO

```bash
python src/fer/fer_yolo.py \
    --input data/input/image.jpg \
    --output data/output/fer_yolo.jpg
```

### FER with MediaPipe

```bash
python src/fer/fer_mediapipe.py \
    --input data/input/image.jpg \
    --output data/output/fer_mediapipe.jpg
```

## Using in Python Code

```python
from src.detection.person_detector import PersonDetector
import cv2

# Initialize detector
detector = PersonDetector(model_path='yolov8n.pt', confidence=0.5)

# Process an image
image = cv2.imread('data/input/image.jpg')
detections = detector.detect_persons(image)
output_image = detector.detect_and_draw(image)

# Save result
cv2.imwrite('data/output/result.jpg', output_image)

print(f"Detected {len(detections)} person(s)")
```

## Troubleshooting

### Issue: "No module named 'cv2'"
**Solution:** Install OpenCV: `pip install opencv-python`

### Issue: "No module named 'ultralytics'"
**Solution:** Install ultralytics: `pip install ultralytics`

### Issue: Model download fails
**Solution:** Check your internet connection. Models are downloaded automatically on first use.

### Issue: CUDA out of memory
**Solution:** Use a smaller model (yolov8n.pt instead of yolov8x.pt) or reduce image resolution.

### Issue: Slow performance
**Solution:** 
- Use GPU if available
- Use smaller models (yolov8n.pt)
- Reduce video resolution
- Process every nth frame instead of all frames

## Next Steps

- Check out the full [README.md](README.md) for detailed documentation
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) to contribute
- Explore the configuration options in [config.yaml](config.yaml)
- Try different YOLO models for better accuracy

## Getting Help

- Open an issue on GitHub
- Check existing issues for solutions
- Read the documentation in README.md

## Useful Commands

```bash
# Update dependencies
pip install --upgrade -r requirements.txt

# Run tests
python -m pytest tests/

# Check installed packages
pip list | grep -E "ultralytics|opencv|mediapipe|torch"

# Deactivate virtual environment
deactivate
```

Happy detecting! 🎯
