# 🎭 Facial Expression Recognition using CNN

Real-time emotion detection system using deep Convolutional Neural Networks (CNN) with webcam support.

## 🌟 Features

- **7 Emotion Classifications**: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- **Real-time Detection**: Process webcam feed in real-time
- **Pre-trained Model**: Includes trained CNN model
- **Interactive Visualization**: Color-coded emotions and probability bars
- **Recording Support**: Save processed video with detections
- **Screenshot Capture**: Capture frames during detection

## 📁 Project Structure

```
FER_using_CNN/
├── src/
│   ├── detector.py              # Real-time emotion detector
│   └── training_the_CNN.ipynb   # Model training notebook
├── models/
│   └── emotion_recognition_model.h5  # Pre-trained model
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Installation

```bash
cd FER_using_CNN
pip install -r requirements.txt
```

### 2. Run Real-time Detection

**Basic webcam detection:**
```bash
python src/detector.py
```

**With emotion probability bars:**
```bash
python src/detector.py --bars
```

**Save recording:**
```bash
python src/detector.py --save --output fer_recording.mp4
```

**Use different camera:**
```bash
python src/detector.py --camera 1
```

**Custom model path:**
```bash
python src/detector.py --model path/to/your/model.h5
```

## 🎮 Controls

During real-time detection:
- **'q' or ESC**: Quit the application
- **'s'**: Toggle emotion probability bars on/off
- **'c'**: Capture screenshot of current frame

## 🎨 Emotion Colors

Each emotion is displayed with a unique color:
- 😠 **Angry**: Red
- 🤢 **Disgust**: Dark Green
- 😨 **Fear**: Purple
- 😊 **Happy**: Green
- 😢 **Sad**: Blue
- 😲 **Surprise**: Yellow
- 😐 **Neutral**: Gray

## 🧠 Model Architecture

Deep CNN with:
- **4 Convolutional Blocks** (64, 128, 256, 512 filters)
- **Batch Normalization** for stable training
- **Dropout Layers** to prevent overfitting
- **Dense Layers** (512, 256 neurons)
- **Softmax Output** for 7 emotion classes

**Input**: 48x48 grayscale images  
**Output**: Probability distribution over 7 emotions

## 📊 Model Performance

Typical performance on FER-2013 dataset:
- **Overall Accuracy**: ~75-85%
- **Best Performance**: Happy, Neutral (80-90%)
- **Moderate Performance**: Angry, Sad (70-80%)
- **Challenging**: Fear, Disgust (60-70%)

## 🔧 Command-Line Arguments

```
usage: detector.py [-h] [--model MODEL] [--camera CAMERA] 
                   [--output OUTPUT] [--save] [--bars]

optional arguments:
  -h, --help            Show help message
  --model, -m MODEL     Path to trained model (default: ../models/emotion_recognition_model.h5)
  --camera, -c CAMERA   Camera device ID (default: 0)
  --output, -o OUTPUT   Output video path
  --save                Save output video
  --bars, -b            Show emotion probability bars
```

## 💻 Python API Usage

You can also use the FERDetector class in your own code:

```python
from src.detector import FERDetector

# Initialize detector
detector = FERDetector(model_path='models/emotion_recognition_model.h5')

# Process webcam
detector.process_webcam(camera_id=0, show_bars=True)

# Or process individual frames
import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if ret:
    output_frame, num_faces = detector.process_frame(frame, show_bars=False)
    cv2.imshow('Result', output_frame)
cap.release()
```

## 📚 Training Your Own Model

To train a custom model:

1. **Prepare your dataset** in the following structure:
```
dataset/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── sad/
│   ├── surprise/
│   └── neutral/
└── test/
    ├── angry/
    ├── disgust/
    └── ... (same structure)
```

2. **Open the training notebook**:
```bash
jupyter notebook src/training_the_CNN.ipynb
```

3. **Follow the notebook** to train your model

4. **Use the trained model** with the detector

## 🎯 Use Cases

- **Customer Experience**: Analyze customer emotions in retail
- **Mental Health**: Monitor emotional states
- **Education**: Assess student engagement
- **Security**: Detect suspicious behavior
- **Entertainment**: Interactive games and applications
- **Research**: Study human emotions and expressions

## 🔍 How It Works

1. **Face Detection**: Uses OpenCV Haar Cascade to detect faces
2. **Preprocessing**: Resizes face to 48x48 and normalizes pixels
3. **Emotion Prediction**: CNN model predicts emotion probabilities
4. **Visualization**: Draws bounding boxes and labels with colors
5. **Real-time Processing**: Continues for each frame

## ⚠️ Troubleshooting

**Camera not opening:**
- Check camera is connected
- Try different camera ID (0, 1, 2, etc.)
- Close other applications using the camera

**Model not loading:**
- Verify model file path is correct
- Ensure model file is not corrupted
- Check TensorFlow is properly installed

**Low FPS:**
- Close other applications
- Use a faster computer
- Reduce video resolution

**Poor detection accuracy:**
- Ensure good lighting conditions
- Face the camera directly
- Remove obstructions (glasses, masks, etc.)

## 📈 Improving Performance

To enhance detection:
1. **Better Lighting**: Ensure face is well-lit
2. **Clear View**: Face camera directly
3. **Stable Position**: Minimize head movement
4. **Clean Background**: Reduce visual clutter
5. **Model Fine-tuning**: Train on specific use cases

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Add more emotion classes
- Improve model architecture
- Support multiple faces
- Add emotion tracking over time
- Mobile deployment

## 📝 License

See LICENSE file for details.

## 🙏 Acknowledgments

- **FER-2013 Dataset**: Training data
- **TensorFlow/Keras**: Deep learning framework
- **OpenCV**: Computer vision library
- **Haar Cascade**: Face detection

---

**Made with ❤️ for Emotion Recognition Applications**
