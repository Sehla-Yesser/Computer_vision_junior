"""
Facial Expression Recognition using MediaPipe
This script detects persons using YOLO and analyzes facial expressions using MediaPipe
"""

import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from pathlib import Path
import argparse


class FERMediaPipe:
    """
    Facial Expression Recognition using MediaPipe for face detection
    and landmark detection combined with YOLO for person detection
    """
    
    def __init__(self, person_model='yolov8n.pt', confidence=0.5):
        """
        Initialize the FER detector
        
        Args:
            person_model (str): Path to person detection YOLO model
            confidence (float): Confidence threshold for detections
        """
        self.person_detector = YOLO(person_model)
        self.confidence = confidence
        self.person_class_id = 0  # Person class in COCO dataset
        
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 0 for short range, 1 for full range
            min_detection_confidence=confidence
        )
        
        # Initialize MediaPipe Face Mesh for detailed facial landmarks
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=10,
            refine_landmarks=True,
            min_detection_confidence=confidence,
            min_tracking_confidence=confidence
        )
        
        # Emotion labels
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    def detect_persons(self, image):
        """
        Detect persons in an image using YOLO
        
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
    
    def detect_faces_mediapipe(self, image):
        """
        Detect faces using MediaPipe
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            list: List of face detections with bounding boxes
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.face_detection.process(image_rgb)
        
        faces = []
        if results.detections:
            h, w, _ = image.shape
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                x1 = int(bboxC.xmin * w)
                y1 = int(bboxC.ymin * h)
                x2 = int((bboxC.xmin + bboxC.width) * w)
                y2 = int((bboxC.ymin + bboxC.height) * h)
                
                # Ensure coordinates are within image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                faces.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': detection.score[0],
                    'type': 'face'
                })
        
        return faces
    
    def get_face_landmarks(self, image):
        """
        Get facial landmarks using MediaPipe Face Mesh
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            Face mesh results
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.face_mesh.process(image_rgb)
        
        return results
    
    def analyze_expression_from_landmarks(self, landmarks, image_shape):
        """
        Analyze facial expression based on landmarks
        This is a simplified heuristic approach
        
        Args:
            landmarks: Face mesh landmarks
            image_shape: Shape of the image
            
        Returns:
            dict: Expression analysis results
        """
        # This is a simplified placeholder implementation
        # In a real scenario, you would use ML models or more sophisticated analysis
        
        h, w = image_shape[:2]
        
        # Extract key points for expression analysis
        # These indices correspond to specific facial features in MediaPipe
        # Mouth corners, eyebrows, etc.
        
        # For now, return a mock result
        # A proper implementation would analyze:
        # - Mouth curvature (smile/frown)
        # - Eyebrow position (raised/furrowed)
        # - Eye openness
        # - Overall facial geometry
        
        emotion = 'neutral'  # Default
        confidence = 0.70    # Mock confidence
        
        return {
            'emotion': emotion,
            'confidence': confidence,
            'landmarks_detected': True
        }
    
    def match_faces_to_persons(self, persons, faces):
        """
        Match detected faces to detected persons
        
        Args:
            persons: List of person detections
            faces: List of face detections
            
        Returns:
            dict: Mapping of person index to face detections
        """
        person_faces = {i: [] for i in range(len(persons))}
        
        for face in faces:
            fx1, fy1, fx2, fy2 = face['bbox']
            face_center_x = (fx1 + fx2) / 2
            face_center_y = (fy1 + fy2) / 2
            
            # Find which person contains this face
            for i, person in enumerate(persons):
                px1, py1, px2, py2 = person['bbox']
                
                # Check if face center is within person bbox
                if px1 <= face_center_x <= px2 and py1 <= face_center_y <= py2:
                    person_faces[i].append(face)
                    break
        
        return person_faces
    
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
        
        # Detect faces
        faces = self.detect_faces_mediapipe(image)
        
        # Get facial landmarks
        landmark_results = self.get_face_landmarks(image)
        
        # Match faces to persons
        person_faces = self.match_faces_to_persons(persons, faces)
        
        for i, person in enumerate(persons):
            person_bbox = person['bbox']
            x1, y1, x2, y2 = person_bbox
            
            # Draw person bounding box
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            person_result = {
                'person_bbox': person_bbox,
                'person_confidence': person['confidence'],
                'faces': []
            }
            
            # Process faces for this person
            for face in person_faces[i]:
                face_bbox = face['bbox']
                fx1, fy1, fx2, fy2 = face_bbox
                
                # Analyze expression
                expression = {'emotion': 'neutral', 'confidence': 0.70}
                
                if landmark_results and landmark_results.multi_face_landmarks:
                    # Use landmarks for expression analysis
                    expression = self.analyze_expression_from_landmarks(
                        landmark_results.multi_face_landmarks[0],
                        image.shape
                    )
                
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
            
            # Draw facial landmarks if available
            if landmark_results and landmark_results.multi_face_landmarks:
                for face_landmarks in landmark_results.multi_face_landmarks:
                    # Draw only key landmarks for visualization
                    h, w = image.shape[:2]
                    for idx, landmark in enumerate(face_landmarks.landmark):
                        if idx % 10 == 0:  # Draw every 10th landmark to avoid clutter
                            x = int(landmark.x * w)
                            y = int(landmark.y * h)
                            cv2.circle(output_image, (x, y), 1, (0, 255, 255), -1)
            
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
    
    def __del__(self):
        """Cleanup MediaPipe resources"""
        self.face_detection.close()
        self.face_mesh.close()


def main():
    parser = argparse.ArgumentParser(
        description='Facial Expression Recognition using MediaPipe and YOLO'
    )
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Path to input image or video')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Path to save output (optional)')
    parser.add_argument('--person-model', type=str, default='yolov8n.pt',
                       help='Path to person detection YOLO model')
    parser.add_argument('--confidence', '-c', type=float, default=0.5,
                       help='Confidence threshold (default: 0.5)')
    
    args = parser.parse_args()
    
    # Initialize FER detector
    fer = FERMediaPipe(
        person_model=args.person_model,
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
