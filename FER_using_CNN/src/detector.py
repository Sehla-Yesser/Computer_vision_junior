"""
Real-time Facial Expression Recognition using CNN
This script uses a trained CNN model to detect emotions from webcam feed
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import argparse
from pathlib import Path
import os


class FERDetector:
    """
    Facial Expression Recognition Detector for real-time emotion detection
    
    This class:
    - Loads pre-trained emotion recognition CNN model
    - Detects faces using OpenCV Haar Cascade
    - Predicts emotions from detected faces
    - Displays results in real-time
    """
    
    def __init__(self, model_path, confidence=0.5):
        """
        Initialize the FER detector
        
        Args:
            model_path (str): Path to the trained emotion recognition model (.h5)
            confidence (float): Confidence threshold for face detection
        """
        print("=" * 70)
        print("🎭 Facial Expression Recognition - Real-time Detector")
        print("=" * 70)
        
        # Load the trained emotion recognition model
        print(f"\n📦 Loading emotion recognition model...")
        try:
            self.model = load_model(model_path)
            print(f"✓ Model loaded successfully from: {model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
        
        # Emotion labels (must match training order)
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        
        # Emotion colors for visualization (BGR format)
        self.emotion_colors = {
            'Angry': (0, 0, 255),      # Red
            'Disgust': (0, 128, 0),    # Dark Green
            'Fear': (128, 0, 128),     # Purple
            'Happy': (0, 255, 0),      # Green
            'Sad': (255, 0, 0),        # Blue
            'Surprise': (0, 255, 255), # Yellow
            'Neutral': (128, 128, 128) # Gray
        }
        
        # Load OpenCV face detector (Haar Cascade)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            print("❌ Error loading face cascade classifier!")
            raise Exception("Failed to load Haar Cascade")
        
        print("✓ Face detector initialized")
        
        self.confidence = confidence
        self.img_size = 48  # Model input size
        
        print(f"\n⚙️  Configuration:")
        print(f"  • Image Size: {self.img_size}x{self.img_size}")
        print(f"  • Number of Emotions: {len(self.emotions)}")
        print(f"  • Emotions: {', '.join(self.emotions)}")
        print("=" * 70)
    
    def detect_faces(self, frame):
        """
        Detect faces in a frame using Haar Cascade
        
        Args:
            frame: Input frame (BGR image)
            
        Returns:
            list: List of face coordinates (x, y, w, h)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return faces
    
    def preprocess_face(self, face_img):
        """
        Preprocess face image for model input
        
        Args:
            face_img: Face image (grayscale)
            
        Returns:
            Preprocessed face image ready for model
        """
        # Resize to model input size
        face_img = cv2.resize(face_img, (self.img_size, self.img_size))
        
        # Normalize pixel values to [0, 1]
        face_img = face_img / 255.0
        
        # Reshape for model input (batch_size, height, width, channels)
        face_img = face_img.reshape(1, self.img_size, self.img_size, 1)
        
        return face_img
    
    def predict_emotion(self, face_img):
        """
        Predict emotion from face image
        
        Args:
            face_img: Preprocessed face image
            
        Returns:
            tuple: (emotion_label, confidence_score)
        """
        # Get prediction
        predictions = self.model.predict(face_img, verbose=0)
        
        # Get emotion with highest confidence
        emotion_idx = np.argmax(predictions[0])
        confidence = predictions[0][emotion_idx]
        emotion = self.emotions[emotion_idx]
        
        return emotion, confidence, predictions[0]
    
    def draw_emotion(self, frame, face_coords, emotion, confidence, all_predictions=None):
        """
        Draw bounding box and emotion label on frame
        
        Args:
            frame: Input frame
            face_coords: Face coordinates (x, y, w, h)
            emotion: Predicted emotion
            confidence: Confidence score
            all_predictions: All emotion predictions (optional)
        """
        x, y, w, h = face_coords
        
        # Get color for this emotion
        color = self.emotion_colors.get(emotion, (255, 255, 255))
        
        # Draw face bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Prepare label text
        label = f"{emotion}: {confidence*100:.1f}%"
        
        # Get text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        
        # Draw background rectangle for text
        cv2.rectangle(
            frame,
            (x, y - text_height - 10),
            (x + text_width, y),
            color,
            -1
        )
        
        # Draw text
        cv2.putText(
            frame,
            label,
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        # Draw emotion bar chart (optional)
        if all_predictions is not None:
            self.draw_emotion_bars(frame, all_predictions, x, y, w)
    
    def draw_emotion_bars(self, frame, predictions, x, y, w):
        """
        Draw emotion probability bars
        
        Args:
            frame: Input frame
            predictions: Array of emotion probabilities
            x, y, w: Face bounding box coordinates
        """
        bar_height = 15
        bar_width = w
        start_y = y - 120
        
        if start_y < 0:
            start_y = y + 170
        
        for i, (emotion, prob) in enumerate(zip(self.emotions, predictions)):
            bar_y = start_y + i * (bar_height + 2)
            
            # Skip if outside frame
            if bar_y < 0 or bar_y > frame.shape[0]:
                continue
            
            # Draw background bar
            cv2.rectangle(
                frame,
                (x, bar_y),
                (x + bar_width, bar_y + bar_height),
                (50, 50, 50),
                -1
            )
            
            # Draw probability bar
            filled_width = int(bar_width * prob)
            color = self.emotion_colors.get(emotion, (255, 255, 255))
            cv2.rectangle(
                frame,
                (x, bar_y),
                (x + filled_width, bar_y + bar_height),
                color,
                -1
            )
            
            # Draw emotion label
            cv2.putText(
                frame,
                f"{emotion}",
                (x + 5, bar_y + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (255, 255, 255),
                1
            )
    
    def process_frame(self, frame, show_bars=False):
        """
        Process a single frame for emotion detection
        
        Args:
            frame: Input frame (BGR image)
            show_bars: Whether to show emotion probability bars
            
        Returns:
            Processed frame with annotations
        """
        # Detect faces
        faces = self.detect_faces(frame)
        
        # Convert to grayscale for emotion detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]
            
            try:
                # Preprocess face
                face_input = self.preprocess_face(face_roi)
                
                # Predict emotion
                emotion, confidence, all_preds = self.predict_emotion(face_input)
                
                # Draw on frame
                if show_bars:
                    self.draw_emotion((x, y, w, h), emotion, confidence, all_preds)
                else:
                    self.draw_emotion(frame, (x, y, w, h), emotion, confidence)
                
            except Exception as e:
                print(f"Error processing face: {e}")
                continue
        
        return frame, len(faces)
    
    def process_webcam(self, camera_id=0, save_output=False, output_path=None, show_bars=False):
        """
        Process real-time webcam feed for emotion detection
        
        Args:
            camera_id (int): Camera device ID
            save_output (bool): Whether to save output video
            output_path (str): Path to save output video
            show_bars (bool): Whether to show emotion probability bars
        """
        # Open webcam
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"❌ Error: Could not open camera {camera_id}")
            return
        
        # Get camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 30
        
        # Setup video writer if saving
        out = None
        if save_output and output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"📹 Recording to: {output_path}")
        
        print("\n▶️  Starting webcam...")
        print("Controls:")
        print("  • Press 'q' or 'ESC' to quit")
        print("  • Press 's' to toggle emotion bars")
        print("  • Press 'c' to capture screenshot")
        print("-" * 70)
        
        frame_count = 0
        total_faces = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Could not read frame")
                break
            
            # Process frame
            output_frame, num_faces = self.process_frame(frame, show_bars=show_bars)
            
            frame_count += 1
            total_faces += num_faces
            
            # Add info overlay
            info_text = f"Frame: {frame_count} | Faces: {num_faces}"
            if frame_count > 0:
                avg_faces = total_faces / frame_count
                info_text += f" | Avg: {avg_faces:.2f}"
            
            cv2.putText(
                output_frame,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # Display instructions
            cv2.putText(
                output_frame,
                "Press 'q' to quit | 's' for bars | 'c' for screenshot",
                (10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            
            # Display frame
            cv2.imshow('Facial Expression Recognition - Real-time', output_frame)
            
            # Save frame if recording
            if out:
                out.write(output_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord('s'):  # Toggle emotion bars
                show_bars = not show_bars
                print(f"Emotion bars: {'ON' if show_bars else 'OFF'}")
            elif key == ord('c'):  # Capture screenshot
                screenshot_path = f'fer_screenshot_{frame_count}.jpg'
                cv2.imwrite(screenshot_path, output_frame)
                print(f"📸 Screenshot saved: {screenshot_path}")
        
        # Cleanup
        cap.release()
        if out:
            out.release()
            print(f"\n✓ Recording saved to: {output_path}")
        cv2.destroyAllWindows()
        
        # Print statistics
        print("\n" + "=" * 70)
        print("📊 Session Statistics:")
        print(f"  • Total frames processed: {frame_count}")
        print(f"  • Total faces detected: {total_faces}")
        if frame_count > 0:
            print(f"  • Average faces per frame: {total_faces/frame_count:.2f}")
        print("=" * 70)


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description='Real-time Facial Expression Recognition using CNN'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='../models/emotion_recognition_model.h5',
        help='Path to trained emotion recognition model'
    )
    parser.add_argument(
        '--camera', '-c',
        type=int,
        default=0,
        help='Camera device ID (default: 0)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Path to save output video (optional)'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save output video'
    )
    parser.add_argument(
        '--bars', '-b',
        action='store_true',
        help='Show emotion probability bars'
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"❌ Error: Model file not found: {args.model}")
        print("\nPlease provide the correct path to the trained model.")
        print("Example: python detector.py --model path/to/model.h5")
        return
    
    # Initialize detector
    try:
        detector = FERDetector(model_path=args.model)
    except Exception as e:
        print(f"❌ Failed to initialize detector: {e}")
        return
    
    # Set output path if saving
    output_path = args.output
    if args.save and output_path is None:
        output_path = 'fer_output.mp4'
    
    # Process webcam
    try:
        detector.process_webcam(
            camera_id=args.camera,
            save_output=args.save or (output_path is not None),
            output_path=output_path,
            show_bars=args.bars
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")


if __name__ == "__main__":
    main()
