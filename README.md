# Computer Vision Junior - Comprehensive Computer Vision Suite

A collection of advanced computer vision models for person detection, facial expression recognition, and driver distraction detection using deep learning.

## 🎯 Project Overview

This repository contains three main computer vision modules designed for real-time analysis:

1. **Person Detector** - Real-time person detection using YOLOv8
2. **Facial Expression Recognition (FER)** - Emotion detection using CNN models
3. **Driver Distraction Detection** - Real-time monitoring of driver attention and behavior

## 📁 Project Structure

```
Computer_vision_junior/
├── person_detector/          # Real-time person detection module
│   ├── src/
│   │   └── detection/
│   │       └── person_detector.py
│   ├── examples/
│   ├── models/
│   └── README.md
│
├── FER_using_CNN/           # Facial Expression Recognition module
│   ├── src/
│   │   └── detector.py
│   ├── models/
│   │   └── emotion_recognition_model.h5
│   └── README.md
│
├── distraction_detection/   # Driver distraction detection (Coming Soon)
│   ├── src/
│   ├── models/
│   └── README.md
│
└── README.md               # This file
```

## 🚀 Modules

### 1. Person Detector

Real-time person detection using YOLOv8 with webcam support.

**Features:**
- ✅ Real-time person detection
- ✅ Multiple YOLO model sizes (nano to extra-large)
- ✅ Live webcam feed processing
- ✅ Recording with detections
- ✅ Customizable confidence thresholds

**Quick Start:**
```bash
cd person_detector
pip install -r requirements.txt
python src/detection/person_detector.py --webcam
```

**Use Cases:**
- Crowd monitoring
- Security surveillance
- People counting
- Social distancing monitoring

[📖 Full Documentation](person_detector/README.md)

---

### 2. Facial Expression Recognition (FER)

Detect and classify facial expressions using deep learning CNN models.

**Features:**
- 🎭 7 emotion classifications (angry, disgust, fear, happy, sad, surprise, neutral)
- 🎥 Real-time webcam processing
- 🧠 CNN-based emotion recognition
- 📊 Confidence scores for predictions
- 💾 Pre-trained model included

**Quick Start:**
```bash
cd FER_using_CNN
pip install -r requirements.txt
python src/detector.py --webcam
```

**Emotions Detected:**
- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

**Use Cases:**
- Customer sentiment analysis
- Mental health monitoring
- Interactive applications
- User experience research
- Education and learning assessment

[📖 Full Documentation](FER_using_CNN/README.md)

---

### 3. Driver Distraction Detection 🚗 (Coming Soon)

Real-time monitoring system to detect driver distraction and improve road safety.

**Planned Features:**
- 👀 Eye gaze tracking
- 📱 Phone usage detection
- 🥱 Drowsiness detection
- 🎯 Head pose estimation
- ⚠️ Real-time alerts
- 📹 Continuous monitoring

**Detection Categories:**
- Looking away from road
- Using mobile phone
- Drowsiness/yawning
- Eating/drinking
- Adjusting radio/controls
- Talking to passengers
- Grooming activities

**Use Cases:**
- Advanced Driver Assistance Systems (ADAS)
- Fleet management
- Driver training
- Insurance telematics
- Road safety research

**Status:** 🔜 In Development

---

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- Webcam or camera device
- (Optional) CUDA-capable GPU for better performance

### General Setup

1. Clone the repository:
```bash
git clone https://github.com/Sehla-Yesser/Computer_vision_junior.git
cd Computer_vision_junior
```

2. Navigate to the specific module:
```bash
cd person_detector  # or FER_using_CNN, or distraction_detection
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the module:
```bash
# Person Detection
python src/detection/person_detector.py --webcam

# Facial Expression Recognition
python src/detector.py --webcam

# Distraction Detection (Coming Soon)
python src/distraction_detector.py --webcam
```

## 📊 Performance

| Module | Model | FPS (CPU) | FPS (GPU) | Accuracy |
|--------|-------|-----------|-----------|----------|
| Person Detector | YOLOv8n | 15-20 | 60-100 | ~90% mAP |
| Person Detector | YOLOv8m | 8-12 | 40-60 | ~95% mAP |
| FER CNN | Custom CNN | 20-30 | 50-80 | ~85% |
| Distraction Detection | TBD | TBD | TBD | TBD |

*Note: Performance varies based on hardware and image resolution*

## 🎮 Usage Examples

### Person Detection
```python
from person_detector.src.detection.person_detector import PersonDetector

detector = PersonDetector(model_path='yolov8n.pt', confidence=0.5)
detector.process_webcam(camera_id=0)
```

### Facial Expression Recognition
```python
from FER_using_CNN.src.detector import FERDetector

fer = FERDetector(model_path='models/emotion_recognition_model.h5')
fer.process_webcam(camera_id=0)
```

### Driver Distraction Detection (Coming Soon)
```python
from distraction_detection.src.distraction_detector import DistractionDetector

detector = DistractionDetector()
detector.process_webcam(camera_id=0)
```

## 🔧 Configuration

Each module can be customized through configuration files or command-line arguments:

- **Confidence Threshold**: Adjust detection sensitivity
- **Model Selection**: Choose between speed and accuracy
- **Camera Source**: Select different camera devices
- **Recording Options**: Save processed video output
- **Display Settings**: Customize visualization

## 📈 Roadmap

- [x] Person Detection Module
- [x] Facial Expression Recognition Module
- [ ] Driver Distraction Detection Module
- [ ] Multi-camera support
- [ ] Cloud deployment options
- [ ] Mobile app integration
- [ ] REST API for remote access
- [ ] Dashboard for analytics
- [ ] Model fine-tuning tools

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

See LICENSE file for details.

## 👥 Authors

**Sehla-Yesser**
- GitHub: [@Sehla-Yesser](https://github.com/Sehla-Yesser)

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - Object detection
- [TensorFlow/Keras](https://www.tensorflow.org/) - Deep learning framework
- [OpenCV](https://opencv.org/) - Computer vision library
- [MediaPipe](https://google.github.io/mediapipe/) - Face detection and landmarks

## 📧 Contact

For questions, suggestions, or collaborations:
- Open an issue on GitHub
- Email: talelboussetta6@gmail.com

## ⚠️ Disclaimer

These models are for educational and research purposes. When deploying in production:
- Ensure proper privacy compliance (GDPR, etc.)
- Obtain necessary permissions for camera usage
- Consider ethical implications of surveillance
- Test thoroughly in your specific use case

---

**Made with ❤️ for Computer Vision Applications**
