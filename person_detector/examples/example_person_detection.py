"""
Example: Person Detection
This example demonstrates how to detect persons using webcam
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection.person_detector import PersonDetector
import cv2


def example_detect_webcam():
    """Example: Detect persons in real-time webcam"""
    print("Example 1: Person Detection - Webcam")
    print("-" * 50)
    print("Press 'q' to quit")
    
    # Initialize the detector
    detector = PersonDetector(model_path='yolov8n.pt', confidence=0.5)
    
    # Process webcam feed
    detector.process_webcam(camera_id=0, save_output=False)


def example_detect_webcam_save():
    """Example: Detect persons in webcam and save output"""
    print("\nExample 2: Person Detection - Webcam (Save Output)")
    print("-" * 50)
    print("Press 'q' to quit")
    
    # Initialize the detector
    detector = PersonDetector(model_path='yolov8n.pt', confidence=0.5)
    
    # Output path
    output_path = 'data/output/person_detection_webcam.mp4'
    
    # Process webcam feed and save
    detector.process_webcam(camera_id=0, save_output=True, output_path=output_path)


def example_detect_webcam_custom():
    """Example: Detect persons with custom annotations"""
    print("\nExample 3: Custom Webcam Detection")
    print("-" * 50)
    print("Press 'q' to quit")
    
    # Initialize the detector
    detector = PersonDetector(model_path='yolov8n.pt', confidence=0.5)
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Webcam started...")
    frame_count = 0
    total_detections = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect persons
        detections = detector.detect_persons(frame)
        
        # Custom drawing
        output_frame = frame.copy()
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            
            # Draw bounding box with custom color
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            # Draw custom label
            label = f"Person #{i+1}: {conf:.2%}"
            
            # Add background for text
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            cv2.rectangle(output_frame, (x1, y1 - text_height - 10), 
                         (x1 + text_width, y1), (0, 0, 255), -1)
            
            # Add text
            cv2.putText(output_frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        frame_count += 1
        total_detections += len(detections)
        
        # Add stats overlay
        stats_text = f"Frame: {frame_count} | Persons: {len(detections)} | Avg: {total_detections/frame_count:.2f}"
        cv2.putText(output_frame, stats_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('Custom Person Detection - Webcam', output_frame)
        
        # Check for 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\nTotal frames: {frame_count}")
    print(f"Total detections: {total_detections}")
    print(f"Average persons per frame: {total_detections/frame_count:.2f}")


if __name__ == "__main__":
    print("="*50)
    print("Person Detection - Webcam Examples")
    print("="*50)
    print("\nAll examples use real-time webcam feed")
    print("Press 'q' in any window to quit\n")
    
    # Create output directory
    Path('../data/output').mkdir(parents=True, exist_ok=True)
    
    # Menu
    print("Available Examples:")
    print("1. Basic Webcam Detection")
    print("2. Webcam Detection (Save Output)")
    print("3. Custom Webcam Detection with Stats")
    print("0. Run all examples")
    
    choice = input("\nSelect example (0-3): ").strip()
    
    try:
        if choice == "1":
            example_detect_webcam()
        elif choice == "2":
            example_detect_webcam_save()
        elif choice == "3":
            example_detect_webcam_custom()
        elif choice == "0":
            print("\nRunning all examples sequentially...")
            print("Close each window to proceed to the next example\n")
            
            try:
                example_detect_webcam()
            except Exception as e:
                print(f"Error in example 1: {e}")
            
            try:
                example_detect_webcam_save()
            except Exception as e:
                print(f"Error in example 2: {e}")
            
            try:
                example_detect_webcam_custom()
            except Exception as e:
                print(f"Error in example 3: {e}")
        else:
            print("Invalid choice!")
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
    
    print("\n" + "="*50)
    print("Examples complete!")
    print("="*50)
