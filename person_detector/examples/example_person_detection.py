"""
Example: Person Detection
This example demonstrates how to detect persons in images and videos
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection.person_detector import PersonDetector
import cv2


def example_detect_in_image():
    """Example: Detect persons in a single image"""
    print("Example 1: Person Detection in Image")
    print("-" * 50)
    
    # Initialize the detector
    detector = PersonDetector(model_path='yolov8n.pt', confidence=0.5)
    
    # Path to input image
    input_path = 'data/input/sample.jpg'
    output_path = 'data/output/example_person_detection.jpg'
    
    # Check if input exists
    if not Path(input_path).exists():
        print(f"Input image not found: {input_path}")
        print("Please place an image at data/input/sample.jpg")
        return
    
    # Process the image
    num_persons = detector.process_image(input_path, output_path)
    
    print(f"Processing complete! Found {num_persons} person(s)")
    print(f"Output saved to: {output_path}")


def example_detect_in_video():
    """Example: Detect persons in a video"""
    print("\nExample 2: Person Detection in Video")
    print("-" * 50)
    
    # Initialize the detector
    detector = PersonDetector(model_path='yolov8n.pt', confidence=0.5)
    
    # Path to input video
    input_path = 'data/input/sample.mp4'
    output_path = 'data/output/example_person_detection.mp4'
    
    # Check if input exists
    if not Path(input_path).exists():
        print(f"Input video not found: {input_path}")
        print("Please place a video at data/input/sample.mp4")
        return
    
    # Process the video
    detector.process_video(input_path, output_path)
    
    print(f"Processing complete!")
    print(f"Output saved to: {output_path}")


def example_detect_with_custom_drawing():
    """Example: Detect persons and draw custom annotations"""
    print("\nExample 3: Custom Annotation")
    print("-" * 50)
    
    # Initialize the detector
    detector = PersonDetector(model_path='yolov8n.pt', confidence=0.5)
    
    input_path = 'data/input/sample.jpg'
    
    if not Path(input_path).exists():
        print(f"Input image not found: {input_path}")
        return
    
    # Read image
    image = cv2.imread(input_path)
    
    # Detect persons
    detections = detector.detect_persons(image)
    
    # Custom drawing
    output_image = image.copy()
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det['bbox']
        conf = det['confidence']
        
        # Draw bounding box with custom color
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 0, 255), 3)
        
        # Draw custom label
        label = f"Person #{i+1}: {conf:.2%}"
        
        # Add background for text
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        cv2.rectangle(output_image, (x1, y1 - text_height - 10), 
                     (x1 + text_width, y1), (0, 0, 255), -1)
        
        # Add text
        cv2.putText(output_image, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Save result
    output_path = 'data/output/example_custom_drawing.jpg'
    cv2.imwrite(output_path, output_image)
    
    print(f"Detected {len(detections)} person(s)")
    print(f"Output saved to: {output_path}")


def example_batch_processing():
    """Example: Process multiple images in a directory"""
    print("\nExample 4: Batch Processing")
    print("-" * 50)
    
    # Initialize the detector
    detector = PersonDetector(model_path='yolov8n.pt', confidence=0.5)
    
    # Get all images from input directory
    input_dir = Path('data/input')
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    images = []
    for ext in image_extensions:
        images.extend(input_dir.glob(f'*{ext}'))
        images.extend(input_dir.glob(f'*{ext.upper()}'))
    
    if not images:
        print("No images found in data/input/")
        return
    
    print(f"Found {len(images)} image(s) to process")
    
    # Process each image
    for i, img_path in enumerate(images, 1):
        print(f"\nProcessing {i}/{len(images)}: {img_path.name}")
        
        output_path = f'data/output/batch_{img_path.stem}_result.jpg'
        num_persons = detector.process_image(str(img_path), output_path)
        
        print(f"  Found {num_persons} person(s)")
    
    print("\nBatch processing complete!")


if __name__ == "__main__":
    print("="*50)
    print("Person Detection Examples")
    print("="*50)
    
    # Run examples
    try:
        example_detect_in_image()
    except Exception as e:
        print(f"Error in example 1: {e}")
    
    try:
        example_detect_in_video()
    except Exception as e:
        print(f"Error in example 2: {e}")
    
    try:
        example_detect_with_custom_drawing()
    except Exception as e:
        print(f"Error in example 3: {e}")
    
    try:
        example_batch_processing()
    except Exception as e:
        print(f"Error in example 4: {e}")
    
    print("\n" + "="*50)
    print("Examples complete!")
    print("="*50)
