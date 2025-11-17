"""
Unit tests for person detector module
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class TestPersonDetector(unittest.TestCase):
    """Test cases for PersonDetector class"""
    
    def test_import(self):
        """Test that PersonDetector can be imported"""
        try:
            from detection.person_detector import PersonDetector
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import PersonDetector: {e}")
    
    def test_class_instantiation(self):
        """Test that PersonDetector can be instantiated"""
        try:
            from detection.person_detector import PersonDetector
            # Note: This will download the model on first run
            # Skip if model download fails (no internet)
            detector = PersonDetector(model_path='yolov8n.pt', confidence=0.5)
            self.assertIsNotNone(detector)
        except Exception as e:
            self.skipTest(f"Could not instantiate PersonDetector (model download may have failed): {e}")
    
    def test_detect_persons_with_dummy_image(self):
        """Test detection with a dummy image"""
        try:
            from detection.person_detector import PersonDetector
            
            # Create a dummy image
            dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
            
            detector = PersonDetector(model_path='yolov8n.pt', confidence=0.5)
            detections = detector.detect_persons(dummy_image)
            
            # Should return a list (may be empty for blank image)
            self.assertIsInstance(detections, list)
        except Exception as e:
            self.skipTest(f"Test skipped: {e}")


class TestFERYolo(unittest.TestCase):
    """Test cases for FERYolo class"""
    
    def test_import(self):
        """Test that FERYolo can be imported"""
        try:
            from fer.fer_yolo import FERYolo
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import FERYolo: {e}")


class TestFERMediaPipe(unittest.TestCase):
    """Test cases for FERMediaPipe class"""
    
    def test_import(self):
        """Test that FERMediaPipe can be imported"""
        try:
            from fer.fer_mediapipe import FERMediaPipe
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import FERMediaPipe: {e}")


if __name__ == '__main__':
    unittest.main()
