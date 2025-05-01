"""
Unit tests for the people detection module.

Tests the PeopleDetector class and its methods to ensure they work correctly.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import torch

# Add parent directory to path so we can import app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.detection import PeopleDetector


class TestPeopleDetector(unittest.TestCase):
    """Test cases for the PeopleDetector class."""
    
    @patch('app.detection.model.YOLO')
    def test_init(self, mock_yolo):
        """Test initialization of the PeopleDetector class."""
        # Setup mocks
        mock_yolo.return_value = MagicMock()
        
        # Create a detector
        detector = PeopleDetector(
            model_name="yolov8n.pt",
            threshold=0.5,
            device="cpu"
        )
        
        # Check initialization values
        self.assertEqual(detector.model_name, "yolov8n.pt")
        self.assertEqual(detector.threshold, 0.5)
        self.assertEqual(detector.device, "cpu")
        self.assertEqual(detector.person_class_id, 0)  # Person is class 0 in COCO/YOLOv8
        
        # Check that the model was loaded
        mock_yolo.assert_called_once_with("yolov8n.pt")
    
    @patch('app.detection.model.YOLO')
    def test_detect(self, mock_yolo):
        """Test detection functionality."""
        # Setup test image (10x10 random color image)
        test_image = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        
        # Setup mocks for YOLOv8 results
        mock_boxes = MagicMock()
        
        # Create two detection results (two people)
        box1 = MagicMock()
        box1.cls = torch.tensor([0.0])  # Person class
        box1.conf = torch.tensor([0.9])  # Confidence
        box1.xyxy = torch.tensor([[10.0, 20.0, 30.0, 40.0]])  # Bounding box
        
        box2 = MagicMock()
        box2.cls = torch.tensor([0.0])  # Person class
        box2.conf = torch.tensor([0.8])  # Confidence
        box2.xyxy = torch.tensor([[50.0, 60.0, 70.0, 80.0]])  # Bounding box
        
        mock_boxes.return_value = [box1, box2]
        
        # Create a mock result object
        mock_result = MagicMock()
        mock_result.boxes = [box1, box2]
        
        # Setup YOLO model mock
        mock_model = MagicMock()
        mock_model.return_value = [mock_result]
        mock_yolo.return_value = mock_model
        
        # Create detector
        detector = PeopleDetector(threshold=0.5, device="cpu")
        
        # Run detection
        detections, inference_time = detector.detect(test_image)
        
        # Check results
        self.assertEqual(len(detections), 2)  # Two people detected
        self.assertEqual(detections[0]["label"], "person")
        self.assertAlmostEqual(detections[0]["score"], 0.9, places=5)
        self.assertEqual(detections[0]["box"], (10, 20, 30, 40))
        self.assertEqual(detections[1]["label"], "person")
        self.assertAlmostEqual(detections[1]["score"], 0.8, places=5)
        self.assertEqual(detections[1]["box"], (50, 60, 70, 80))
        self.assertIsInstance(inference_time, float)
        
        # Check that model was called with correct parameters
        mock_model.assert_called_once_with(test_image, conf=0.5, device="cpu")
    
    @patch('app.detection.model.YOLO')
    def test_update_threshold(self, mock_yolo):
        """Test updating the detection threshold."""
        # Setup mocks
        mock_yolo.return_value = MagicMock()
        
        # Create detector
        detector = PeopleDetector(threshold=0.5)
        
        # Update threshold
        detector.update_threshold(0.8)
        
        # Check that threshold was updated
        self.assertEqual(detector.threshold, 0.8)


if __name__ == '__main__':
    unittest.main() 