"""
Person Detection using YOLOv8 from Ultralytics
This script detects persons in images or videos using YOLO
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import argparse


class PersonDetector:
    """
    A class to handle person detection using YOLO
    """
    
    def __init__(self, model_path='yolov8n.pt', confidence=0.5):
        """
        Initialize the person detector
        
        Args:
            model_path (str): Path to YOLO model weights
            confidence (float): Confidence threshold for detections
        """
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.person_class_id = 0  # Person class in COCO dataset
        
    def detect_persons(self, image):
        """
        Detect persons in an image
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            list: List of detections with bounding boxes and confidence scores
        """
        results = self.model(image, conf=self.confidence, classes=[self.person_class_id])
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(conf)
                })
        
        return detections
    
    def detect_and_draw(self, image):
        """
        Detect persons and draw bounding boxes on image
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            numpy.ndarray: Image with drawn bounding boxes
        """
        detections = self.detect_persons(image)
        output_image = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            
            # Draw bounding box
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"Person: {conf:.2f}"
            cv2.putText(output_image, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return output_image
    
    def process_image(self, input_path, output_path=None):
        """
        Process a single image
        
        Args:
            input_path (str): Path to input image
            output_path (str): Path to save output image (optional)
            
        Returns:
            int: Number of persons detected
        """
        image = cv2.imread(input_path)
        if image is None:
            print(f"Error: Could not read image from {input_path}")
            return 0
        
        detections = self.detect_persons(image)
        output_image = self.detect_and_draw(image)
        
        print(f"Detected {len(detections)} person(s)")
        
        if output_path:
            cv2.imwrite(output_path, output_image)
            print(f"Output saved to {output_path}")
        
        return len(detections)
    
    def process_video(self, input_path, output_path=None):
        """
        Process a video file
        
        Args:
            input_path (str): Path to input video
            output_path (str): Path to save output video (optional)
        """
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {input_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer if output path is provided
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        total_detections = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            detections = self.detect_persons(frame)
            output_frame = self.detect_and_draw(frame)
            
            frame_count += 1
            total_detections += len(detections)
            
            if out:
                out.write(output_frame)
            
            # Display progress
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames, "
                      f"Average persons per frame: {total_detections / frame_count:.2f}")
        
        cap.release()
        if out:
            out.release()
            print(f"Output video saved to {output_path}")
        
        print(f"Total frames: {frame_count}")
        print(f"Total detections: {total_detections}")
        print(f"Average persons per frame: {total_detections / frame_count:.2f}")
    
    def process_webcam(self, camera_id=0, save_output=False, output_path=None):
        """
        Process real-time webcam feed
        
        Args:
            camera_id (int): Camera device ID (default: 0)
            save_output (bool): Whether to save the output video
            output_path (str): Path to save output video (optional)
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return
        
        # Get camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 30  # Default fps for webcam
        
        # Setup video writer if saving is enabled
        out = None
        if save_output and output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print("Starting webcam... Press 'q' to quit")
        frame_count = 0
        total_detections = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            detections = self.detect_persons(frame)
            output_frame = self.detect_and_draw(frame)
            
            frame_count += 1
            total_detections += len(detections)
            
            # Add FPS and detection count to frame
            fps_text = f"FPS: {fps} | Persons: {len(detections)}"
            cv2.putText(output_frame, fps_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow('Person Detection - Webcam', output_frame)
            
            if out:
                out.write(output_frame)
            
            # Check for 'q' key to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        if out:
            out.release()
            print(f"Output video saved to {output_path}")
        cv2.destroyAllWindows()
        
        print(f"\nSession Statistics:")
        print(f"Total frames processed: {frame_count}")
        print(f"Total detections: {total_detections}")
        if frame_count > 0:
            print(f"Average persons per frame: {total_detections / frame_count:.2f}")


def main():
    parser = argparse.ArgumentParser(description='Detect persons in images, videos, or webcam using YOLO')
    parser.add_argument('--input', '-i', type=str, default=None,
                       help='Path to input image or video (omit for webcam)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Path to save output (optional)')
    parser.add_argument('--model', '-m', type=str, default='yolov8n.pt',
                       help='Path to YOLO model (default: yolov8n.pt)')
    parser.add_argument('--confidence', '-c', type=float, default=0.5,
                       help='Confidence threshold (default: 0.5)')
    parser.add_argument('--webcam', '-w', action='store_true',
                       help='Use webcam for real-time detection')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID (default: 0)')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = PersonDetector(model_path=args.model, confidence=args.confidence)
    
    # Use webcam if --webcam flag is set or no input is provided
    if args.webcam or args.input is None:
        save_output = args.output is not None
        detector.process_webcam(camera_id=args.camera, save_output=save_output, output_path=args.output)
    else:
        # Check if input is image or video
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input file {args.input} does not exist")
            return
        
        # Process based on file extension
        if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            detector.process_image(args.input, args.output)
        elif input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
            detector.process_video(args.input, args.output)
        else:
            print(f"Error: Unsupported file format {input_path.suffix}")


if __name__ == "__main__":
    main()
