# Examples

This directory contains example scripts demonstrating how to use the Person Detector package.

## Available Examples

### 1. Person Detection Examples (`example_person_detection.py`)

Demonstrates various ways to detect persons in images and videos:

- **Example 1**: Basic person detection in a single image
- **Example 2**: Person detection in a video
- **Example 3**: Custom annotation and drawing
- **Example 4**: Batch processing multiple images

**Run:**
```bash
python examples/example_person_detection.py
```

### 2. Facial Expression Recognition Examples (`example_fer.py`)

Demonstrates FER using both YOLO and MediaPipe approaches:

- **Example 1**: FER using YOLO
- **Example 2**: FER using MediaPipe
- **Example 3**: Compare YOLO vs MediaPipe methods
- **Example 4**: FER on video
- **Example 5**: Custom FER analysis with detailed output

**Run:**
```bash
python examples/example_fer.py
```

## Before Running Examples

### 1. Install Dependencies

Make sure you have installed all required packages:

```bash
pip install -r requirements.txt
```

### 2. Check Your Webcam

- Ensure your webcam is connected and working
- Close any other applications using the webcam
- The examples will use camera ID 0 by default (your primary webcam)

### 3. Run the Examples

From the examples directory:

```bash
cd examples
python example_person_detection.py
```

Or from the project root:

```bash
python examples/example_person_detection.py
```

## Output

- **Live Display**: See detection results in real-time on your screen
- **Saved Recordings**: If enabled, recordings are saved to `data/output/` directory:
  - `person_detection_webcam.mp4`
  - `fer_yolo_webcam.mp4`
  - `fer_mediapipe_webcam.mp4`

## Controls

- **Press 'q'**: Quit and close the window
- Each example runs until you press 'q'

## Modifying Examples

Feel free to modify these examples to suit your needs:

1. Change confidence thresholds
2. Use different YOLO models (yolov8s.pt, yolov8m.pt, etc.)
3. Adjust drawing colors and styles
4. Add custom processing logic
5. Change camera source (use different camera ID)

## Common Issues

### "Could not open camera"
**Solution:** 
- Check if webcam is connected
- Close other applications using the webcam
- Try different camera IDs (0, 1, 2, etc.)

### "No module named 'cv2'"
**Solution:** Install OpenCV: `pip install opencv-python`

### "Slow frame rate"
**Solution:** Use a faster YOLO model (yolov8n.pt) or enable GPU acceleration

## Creating Your Own Examples

To create your own example:

1. Copy one of the existing example files
2. Modify the code to suit your use case
3. Add appropriate documentation
4. Test thoroughly
5. Share with the community via a pull request!

## Need Help?

- Check the main [README.md](../README.md) for detailed documentation
- Review the [QUICKSTART.md](../QUICKSTART.md) guide
- Open an issue on GitHub
