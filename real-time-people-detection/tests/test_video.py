"""
Unit tests for the video utilities module.

Tests the VideoSource class and various video processing functions.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import cv2

# Add parent directory to path so we can import app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.utils.video import VideoSource, draw_detections, add_performance_stats


class TestVideoSource(unittest.TestCase):
    """Test cases for the VideoSource class."""
    
    @patch('cv2.VideoCapture')
    def test_init(self, mock_capture):
        """Test initialization of the VideoSource class."""
        # Create a video source
        source = VideoSource(
            source=0,
            width=1280,
            height=720,
            fps_buffer_size=60
        )
        
        # Check initialization values
        self.assertEqual(source.source, 0)
        self.assertEqual(source.width, 1280)
        self.assertEqual(source.height, 720)
        self.assertEqual(source.fps_buffer_size, 60)
        self.assertFalse(source.is_running)
        self.assertEqual(source.frame_times, [])
        self.assertIsNone(source.cap)
        
    @patch('cv2.VideoCapture')
    def test_start(self, mock_capture):
        """Test starting the video capture."""
        # Setup mock
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = True
        mock_capture.return_value = mock_instance
        
        # Create and start video source
        source = VideoSource(source=0)
        result = source.start()
        
        # Check results
        self.assertTrue(result)
        self.assertTrue(source.is_running)
        self.assertIsNotNone(source.cap)
        
        # Check that properties were set for webcam
        mock_instance.set.assert_any_call(cv2.CAP_PROP_FRAME_WIDTH, 640)
        mock_instance.set.assert_any_call(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    @patch('cv2.VideoCapture')
    def test_start_video_file(self, mock_capture):
        """Test starting the video capture with a file source."""
        # Setup mock
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = True
        mock_capture.return_value = mock_instance
        
        # Create and start video source with a file path
        source = VideoSource(source="test_video.mp4")
        result = source.start()
        
        # Check results
        self.assertTrue(result)
        self.assertTrue(source.is_running)
        
        # Check that properties were NOT set for file source
        mock_instance.set.assert_not_called()
    
    @patch('cv2.VideoCapture')
    def test_start_failure(self, mock_capture):
        """Test handling failure to open the video source."""
        # Setup mock to fail on open
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = False
        mock_capture.return_value = mock_instance
        
        # Create and try to start video source
        source = VideoSource(source=0)
        result = source.start()
        
        # Check results
        self.assertFalse(result)
        self.assertFalse(source.is_running)
    
    @patch('cv2.VideoCapture')
    def test_stop(self, mock_capture):
        """Test stopping the video capture."""
        # Setup mock
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = True
        mock_capture.return_value = mock_instance
        
        # Create and start video source
        source = VideoSource(source=0)
        source.start()
        
        # Stop video source
        source.stop()
        
        # Check results
        self.assertFalse(source.is_running)
        mock_instance.release.assert_called_once()
    
    @patch('cv2.VideoCapture')
    @patch('time.time')
    def test_read_frame(self, mock_time, mock_capture):
        """Test reading frames from the video source."""
        # Setup mocks
        mock_time.return_value = 12345.0
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = True
        mock_instance.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_capture.return_value = mock_instance
        
        # Create, start, and read from video source
        source = VideoSource(source=0)
        source.start()
        ret, frame = source.read_frame()
        
        # Check results
        self.assertTrue(ret)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (480, 640, 3))
        self.assertEqual(source.frame_times, [12345.0])
    
    @patch('cv2.VideoCapture')
    @patch('time.time')
    def test_get_fps(self, mock_time, mock_capture):
        """Test FPS calculation."""
        # Setup mocks
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = True
        mock_capture.return_value = mock_instance
        
        # Create and start video source
        source = VideoSource(source=0)
        source.start()
        
        # Manually set frame times (10 frames over 1 second = 10 FPS)
        source.frame_times = [100.0, 100.1, 100.2, 100.3, 100.4, 100.5, 100.6, 100.7, 100.8, 100.9, 101.0]
        
        # Check FPS calculation
        fps = source.get_fps()
        self.assertAlmostEqual(fps, 10.0, places=1)


class TestDrawingFunctions(unittest.TestCase):
    """Test cases for drawing functions."""
    
    def test_draw_detections(self):
        """Test drawing detection boxes and labels."""
        # Create test image and detections
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [
            {'box': (100, 100, 200, 200), 'score': 0.95, 'label': 'person'},
            {'box': (300, 300, 400, 400), 'score': 0.85, 'label': 'person'}
        ]
        
        # Draw detections
        result = draw_detections(image, detections)
        
        # Check result is a valid image
        self.assertEqual(result.shape, image.shape)
        self.assertFalse(np.array_equal(result, image))  # Should be different from original
    
    def test_add_performance_stats(self):
        """Test adding performance stats to the image."""
        # Create test image
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add stats
        result = add_performance_stats(image, 30.0, 0.05)
        
        # Check result is a valid image
        self.assertEqual(result.shape, image.shape)
        self.assertFalse(np.array_equal(result, image))  # Should be different from original


if __name__ == '__main__':
    unittest.main() 