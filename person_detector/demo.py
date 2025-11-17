"""
Demo script for Person Detection and Facial Expression Recognition

This script demonstrates how to use the person detector and FER modules.
"""

import cv2
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from detection.person_detector import PersonDetector
from fer.fer_yolo import FERYolo
from fer.fer_mediapipe import FERMediaPipe


def create_sample_image():
    """
    Create a sample image with text for testing when no input image is available
    """
    # Create a blank image
    image = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # Add text
    text = "Place an image in data/input/"
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    text_x = (image.shape[1] - text_size[0]) // 2
    text_y = (image.shape[0] + text_size[1]) // 2
    
    cv2.putText(image, text, (text_x, text_y - 40), font, 1, (0, 0, 0), 2)
    cv2.putText(image, "to test person detection", (text_x - 50, text_y + 20), 
                font, 0.8, (0, 0, 0), 2)
    
    return image


def demo_person_detection(input_path=None):
    """
    Demonstrate person detection
    """
    print("\n" + "="*60)
    print("DEMO: Person Detection with YOLO")
    print("="*60)
    
    # Initialize detector
    print("Initializing person detector...")
    detector = PersonDetector(model_path='yolov8n.pt', confidence=0.5)
    
    if input_path and Path(input_path).exists():
        print(f"Processing image: {input_path}")
        detector.process_image(input_path, 'data/output/person_detection_demo.jpg')
    else:
        print("No input image provided. Creating sample image...")
        image = create_sample_image()
        cv2.imwrite('data/output/sample_image.jpg', image)
        print("Sample image saved to: data/output/sample_image.jpg")
        print("Note: This is just a blank image for demonstration.")
        print("Add real images with people to data/input/ for actual detection.")


def demo_fer_yolo(input_path=None):
    """
    Demonstrate FER with YOLO
    """
    print("\n" + "="*60)
    print("DEMO: Facial Expression Recognition with YOLO")
    print("="*60)
    
    # Initialize FER
    print("Initializing FER with YOLO...")
    fer = FERYolo(person_model='yolov8n.pt', confidence=0.5)
    
    if input_path and Path(input_path).exists():
        print(f"Processing image: {input_path}")
        fer.process_image_file(input_path, 'data/output/fer_yolo_demo.jpg')
    else:
        print("No input image provided.")
        print("Add images with people and faces to data/input/ for FER analysis.")


def demo_fer_mediapipe(input_path=None):
    """
    Demonstrate FER with MediaPipe
    """
    print("\n" + "="*60)
    print("DEMO: Facial Expression Recognition with MediaPipe")
    print("="*60)
    
    # Initialize FER
    print("Initializing FER with MediaPipe...")
    fer = FERMediaPipe(person_model='yolov8n.pt', confidence=0.5)
    
    if input_path and Path(input_path).exists():
        print(f"Processing image: {input_path}")
        fer.process_image_file(input_path, 'data/output/fer_mediapipe_demo.jpg')
    else:
        print("No input image provided.")
        print("Add images with people and faces to data/input/ for FER analysis.")


def main():
    """
    Main demo function
    """
    print("\n" + "="*60)
    print("Person Detector - Demo Script")
    print("="*60)
    
    # Create output directory if it doesn't exist
    Path('data/output').mkdir(parents=True, exist_ok=True)
    
    # Check for input images
    input_dir = Path('data/input')
    input_images = list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.png'))
    
    if input_images:
        print(f"\nFound {len(input_images)} image(s) in data/input/")
        input_path = str(input_images[0])
        print(f"Using: {input_path}")
    else:
        print("\nNo images found in data/input/")
        print("The demo will run with limited functionality.")
        input_path = None
    
    # Run demos
    try:
        demo_person_detection(input_path)
    except Exception as e:
        print(f"Error in person detection demo: {e}")
    
    try:
        demo_fer_yolo(input_path)
    except Exception as e:
        print(f"Error in FER YOLO demo: {e}")
    
    try:
        demo_fer_mediapipe(input_path)
    except Exception as e:
        print(f"Error in FER MediaPipe demo: {e}")
    
    print("\n" + "="*60)
    print("Demo completed!")
    print("="*60)
    print("\nTo use with your own images:")
    print("1. Place images in data/input/")
    print("2. Run: python demo.py")
    print("\nFor more options, use the individual scripts:")
    print("  python src/detection/person_detector.py --help")
    print("  python src/fer/fer_yolo.py --help")
    print("  python src/fer/fer_mediapipe.py --help")
    print()


if __name__ == "__main__":
    main()
