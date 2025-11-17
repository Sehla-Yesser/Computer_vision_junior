"""
Facial Expression Recognition Module
"""

from .fer_yolo import FERYolo
from .fer_mediapipe import FERMediaPipe

__all__ = ['FERYolo', 'FERMediaPipe']
