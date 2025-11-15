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

### 2. Prepare Input Data

Place your test images in the `data/input/` directory:

- `sample.jpg` - For person detection examples
- `sample_face.jpg` - For FER examples with faces
- `sample.mp4` or `sample_video.mp4` - For video examples

If you don't have sample images, the scripts will inform you which files are needed.

### 3. Run the Examples

From the project root directory:

```bash
# Person detection examples
python examples/example_person_detection.py

# FER examples
python examples/example_fer.py
```

## Output

All example outputs will be saved to the `data/output/` directory with descriptive filenames:

- `example_person_detection.jpg`
- `example_fer_yolo.jpg`
- `example_fer_mediapipe.jpg`
- `compare_yolo.jpg`
- `compare_mediapipe.jpg`
- `custom_fer_analysis.jpg`
- etc.

## Modifying Examples

Feel free to modify these examples to suit your needs:

1. Change confidence thresholds
2. Use different YOLO models (yolov8s.pt, yolov8m.pt, etc.)
3. Adjust drawing colors and styles
4. Add custom processing logic
5. Export results to different formats

## Common Issues

### "No module named 'src'"
**Solution:** Run the examples from the project root directory, not from the examples directory.

### "Input image not found"
**Solution:** Make sure you have placed the required images in `data/input/` directory.

### "No module named 'cv2'"
**Solution:** Install OpenCV: `pip install opencv-python`

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
