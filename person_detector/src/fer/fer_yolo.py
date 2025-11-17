"""
Facial Expression Recognition using YOLO
This script detects persons and their facial expressions using YOLO models
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import argparse


class FERYolo:
    """
    Facial Expression Recognition using YOLO
    """
    
    def __init__(self, person_model='yolov8n.pt', face_model='yolov8n-face.pt', 
                 confidence=0.5):
        """
        Initialize the FER detector
        
        Args:
            person_model (str): Path to person detection YOLO model
            face_model (str): Path to face detection YOLO model
            confidence (float): Confidence threshold for detections
        """
        self.person_detector = YOLO(person_model)
        self.confidence = confidence
        self.person_class_id = 0  # Person class in COCO dataset
        
        # Note: For FER, you might need a specialized model
        # This is a placeholder that detects faces using YOLO
        try:
            self.face_detector = YOLO(face_model)
            self.has_face_model = True
        except:
            print("Warning: Face detection model not found. Using person detection only.")
            self.has_face_model = False
        
        # Emotion labels (would be used with a proper FER model)
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    def detect_persons(self, image):
        """
        Detect persons in an image
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            list: List of person detections
        """
        results = self.person_detector(image, conf=self.confidence, classes=[self.person_class_id])
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(conf),
                    'type': 'person'
                })
        
        return detections
    
    def detect_faces_in_region(self, image, person_bbox):
        """
        Detect faces within a person's bounding box
        
        Args:
            image: Full image
            person_bbox: Person's bounding box [x1, y1, x2, y2]
            
        Returns:
            list: List of face detections within the person region
        """
        if not self.has_face_model:
            return []
        
        x1, y1, x2, y2 = person_bbox
        person_roi = image[y1:y2, x1:x2]
        
        if person_roi.size == 0:
            return []
        
        # Detect faces in the person ROI
        results = self.face_detector(person_roi, conf=self.confidence)
        
        faces = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                fx1, fy1, fx2, fy2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                
                # Convert coordinates back to full image space
                faces.append({
                    'bbox': [int(x1 + fx1), int(y1 + fy1), 
                            int(x1 + fx2), int(y1 + fy2)],
                    'confidence': float(conf),
                    'type': 'face'
                })
        
        return faces
    
    def analyze_expression(self, image, face_bbox):
        """
        Analyze facial expression
        
        Args:
            image: Input image
            face_bbox: Face bounding box [x1, y1, x2, y2]
            
        Returns:
            dict: Expression analysis results
        """
        # This is a placeholder implementation
        # In a real implementation, you would use a FER model here
        # For now, we return a mock result
        
        x1, y1, x2, y2 = face_bbox
        face_roi = image[y1:y2, x1:x2]
        
        if face_roi.size == 0:
            return {'emotion': 'unknown', 'confidence': 0.0}
        
        # Mock emotion detection (replace with actual FER model)
        emotion = 'neutral'  # Default emotion
        confidence = 0.75    # Mock confidence
        
        return {
            'emotion': emotion,
            'confidence': confidence
        }
    
    def process_image(self, image):
        """
        Process image to detect persons and their facial expressions
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            tuple: (annotated_image, results)
        """
        output_image = image.copy()
        results = []
        
        # Detect persons
        persons = self.detect_persons(image)
        
        for person in persons:
            person_bbox = person['bbox']
            x1, y1, x2, y2 = person_bbox
            
            # Draw person bounding box
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Detect faces in person region
            faces = self.detect_faces_in_region(image, person_bbox)
            
            person_result = {
                'person_bbox': person_bbox,
                'person_confidence': person['confidence'],
                'faces': []
            }
            
            for face in faces:
                face_bbox = face['bbox']
                fx1, fy1, fx2, fy2 = face_bbox
                
                # Analyze expression
                expression = self.analyze_expression(image, face_bbox)
                
                # Draw face bounding box
                cv2.rectangle(output_image, (fx1, fy1), (fx2, fy2), (255, 0, 0), 2)
                
                # Draw emotion label
                label = f"{expression['emotion']}: {expression['confidence']:.2f}"
                cv2.putText(output_image, label, (fx1, fy1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                person_result['faces'].append({
                    'face_bbox': face_bbox,
                    'face_confidence': face['confidence'],
                    'emotion': expression['emotion'],
                    'emotion_confidence': expression['confidence']
                })
            
            # Draw person label
            label = f"Person: {person['confidence']:.2f}"
            if person_result['faces']:
                label += f" ({len(person_result['faces'])} face(s))"
            cv2.putText(output_image, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            results.append(person_result)
        
        return output_image, results
    
    def process_image_file(self, input_path, output_path=None):
        """
        Process a single image file
        
        Args:
            input_path (str): Path to input image
            output_path (str): Path to save output image (optional)
        """
        image = cv2.imread(input_path)
        if image is None:
            print(f"Error: Could not read image from {input_path}")
            return
        
        output_image, results = self.process_image(image)
        
        print(f"\nDetected {len(results)} person(s)")
        for i, result in enumerate(results):
            print(f"Person {i+1}:")
            print(f"  Confidence: {result['person_confidence']:.2f}")
            print(f"  Faces detected: {len(result['faces'])}")
            for j, face in enumerate(result['faces']):
                print(f"    Face {j+1}: {face['emotion']} "
                      f"(confidence: {face['emotion_confidence']:.2f})")
        
        if output_path:
            cv2.imwrite(output_path, output_image)
            print(f"\nOutput saved to {output_path}")
    
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
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            output_frame, results = self.process_image(frame)
            
            frame_count += 1
            
            if out:
                out.write(output_frame)
            
            # Display progress
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...")
        
        cap.release()
        if out:
            out.release()
            print(f"Output video saved to {output_path}")
        
        print(f"Total frames processed: {frame_count}")


def main():
    parser = argparse.ArgumentParser(
        description='Facial Expression Recognition using YOLO'
    )
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Path to input image or video')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Path to save output (optional)')
    parser.add_argument('--person-model', type=str, default='yolov8n.pt',
                       help='Path to person detection YOLO model')
    parser.add_argument('--face-model', type=str, default='yolov8n-face.pt',
                       help='Path to face detection YOLO model')
    parser.add_argument('--confidence', '-c', type=float, default=0.5,
                       help='Confidence threshold (default: 0.5)')
    
    args = parser.parse_args()
    
    # Initialize FER detector
    fer = FERYolo(
        person_model=args.person_model,
        face_model=args.face_model,
        confidence=args.confidence
    )
    
    # Check if input exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file {args.input} does not exist")
        return
    
    # Process based on file extension
    if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
        fer.process_image_file(args.input, args.output)
    elif input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
        fer.process_video(args.input, args.output)
    else:
        print(f"Error: Unsupported file format {input_path.suffix}")


if __name__ == "__main__":
    main()
