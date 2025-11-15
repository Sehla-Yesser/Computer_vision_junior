"""
Example: Facial Expression Recognition
This example demonstrates how to use both YOLO and MediaPipe for FER
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fer.fer_yolo import FERYolo
from src.fer.fer_mediapipe import FERMediaPipe
import cv2


def example_fer_yolo():
    """Example: FER using YOLO"""
    print("Example 1: FER with YOLO")
    print("-" * 50)
    
    # Initialize FER detector
    fer = FERYolo(person_model='yolov8n.pt', confidence=0.5)
    
    # Input and output paths
    input_path = 'data/input/sample_face.jpg'
    output_path = 'data/output/example_fer_yolo.jpg'
    
    # Check if input exists
    if not Path(input_path).exists():
        print(f"Input image not found: {input_path}")
        print("Please place an image with faces at data/input/sample_face.jpg")
        return
    
    # Process the image
    fer.process_image_file(input_path, output_path)
    
    print(f"\nOutput saved to: {output_path}")


def example_fer_mediapipe():
    """Example: FER using MediaPipe"""
    print("\nExample 2: FER with MediaPipe")
    print("-" * 50)
    
    # Initialize FER detector
    fer = FERMediaPipe(person_model='yolov8n.pt', confidence=0.5)
    
    # Input and output paths
    input_path = 'data/input/sample_face.jpg'
    output_path = 'data/output/example_fer_mediapipe.jpg'
    
    # Check if input exists
    if not Path(input_path).exists():
        print(f"Input image not found: {input_path}")
        print("Please place an image with faces at data/input/sample_face.jpg")
        return
    
    # Process the image
    fer.process_image_file(input_path, output_path)
    
    print(f"\nOutput saved to: {output_path}")


def example_compare_methods():
    """Example: Compare YOLO and MediaPipe FER"""
    print("\nExample 3: Compare YOLO vs MediaPipe")
    print("-" * 50)
    
    input_path = 'data/input/sample_face.jpg'
    
    if not Path(input_path).exists():
        print(f"Input image not found: {input_path}")
        return
    
    # Read image
    image = cv2.imread(input_path)
    
    # Process with YOLO
    print("\nProcessing with YOLO...")
    fer_yolo = FERYolo(person_model='yolov8n.pt', confidence=0.5)
    output_yolo, results_yolo = fer_yolo.process_image(image)
    cv2.imwrite('data/output/compare_yolo.jpg', output_yolo)
    
    # Process with MediaPipe
    print("Processing with MediaPipe...")
    fer_mp = FERMediaPipe(person_model='yolov8n.pt', confidence=0.5)
    output_mp, results_mp = fer_mp.process_image(image)
    cv2.imwrite('data/output/compare_mediapipe.jpg', output_mp)
    
    # Compare results
    print("\nComparison Results:")
    print(f"YOLO - Detected {len(results_yolo)} person(s)")
    print(f"MediaPipe - Detected {len(results_mp)} person(s)")
    
    print("\nOutputs saved:")
    print("  - data/output/compare_yolo.jpg")
    print("  - data/output/compare_mediapipe.jpg")


def example_fer_video():
    """Example: FER on video using MediaPipe"""
    print("\nExample 4: FER on Video")
    print("-" * 50)
    
    # Initialize FER detector
    fer = FERMediaPipe(person_model='yolov8n.pt', confidence=0.5)
    
    # Input and output paths
    input_path = 'data/input/sample_video.mp4'
    output_path = 'data/output/example_fer_video.mp4'
    
    # Check if input exists
    if not Path(input_path).exists():
        print(f"Input video not found: {input_path}")
        print("Please place a video at data/input/sample_video.mp4")
        return
    
    # Process the video
    fer.process_video(input_path, output_path)
    
    print(f"\nOutput saved to: {output_path}")


def example_custom_fer_analysis():
    """Example: Custom FER analysis with detailed output"""
    print("\nExample 5: Custom FER Analysis")
    print("-" * 50)
    
    input_path = 'data/input/sample_face.jpg'
    
    if not Path(input_path).exists():
        print(f"Input image not found: {input_path}")
        return
    
    # Read image
    image = cv2.imread(input_path)
    
    # Initialize FER
    fer = FERMediaPipe(person_model='yolov8n.pt', confidence=0.5)
    
    # Process image
    output_image, results = fer.process_image(image)
    
    # Detailed analysis
    print("\nDetailed Analysis:")
    print("="*50)
    
    for i, person_result in enumerate(results, 1):
        print(f"\nPerson {i}:")
        print(f"  Bounding Box: {person_result['person_bbox']}")
        print(f"  Confidence: {person_result['person_confidence']:.2f}")
        print(f"  Number of Faces: {len(person_result['faces'])}")
        
        for j, face in enumerate(person_result['faces'], 1):
            print(f"\n  Face {j}:")
            print(f"    Bounding Box: {face['face_bbox']}")
            print(f"    Face Confidence: {face['face_confidence']:.2f}")
            print(f"    Emotion: {face['emotion']}")
            print(f"    Emotion Confidence: {face['emotion_confidence']:.2f}")
    
    # Save annotated image
    output_path = 'data/output/custom_fer_analysis.jpg'
    cv2.imwrite(output_path, output_image)
    print(f"\nAnnotated image saved to: {output_path}")


if __name__ == "__main__":
    print("="*50)
    print("Facial Expression Recognition Examples")
    print("="*50)
    
    # Create output directory
    Path('data/output').mkdir(parents=True, exist_ok=True)
    
    # Run examples
    try:
        example_fer_yolo()
    except Exception as e:
        print(f"Error in example 1: {e}")
    
    try:
        example_fer_mediapipe()
    except Exception as e:
        print(f"Error in example 2: {e}")
    
    try:
        example_compare_methods()
    except Exception as e:
        print(f"Error in example 3: {e}")
    
    try:
        example_fer_video()
    except Exception as e:
        print(f"Error in example 4: {e}")
    
    try:
        example_custom_fer_analysis()
    except Exception as e:
        print(f"Error in example 5: {e}")
    
    print("\n" + "="*50)
    print("Examples complete!")
    print("="*50)
