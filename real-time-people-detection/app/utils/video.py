"""
Video utilities for capturing and processing video frames.

This module handles video input/output operations, including webcam access,
video file reading, and frame processing.
"""

import time
from typing import Dict, Generator, Optional, Tuple, List, Any, Union

import cv2
import numpy as np


class VideoSource:
    """
    A class for handling video input from different sources (webcam or file).
    
    Attributes:
        source: Camera index (int) or video file path (str)
        width: Frame width to set (if possible)
        height: Frame height to set (if possible)
        fps_buffer_size: Number of frames to average for FPS calculation
    """
    
    def __init__(
        self,
        source: Union[int, str] = 0,
        width: int = 640,
        height: int = 480,
        fps_buffer_size: int = 30,
    ):
        """
        Initialize the video source.
        
        Args:
            source: Camera index (int) or video file path (str)
            width: Width to set for the captured frames
            height: Height to set for the captured frames
            fps_buffer_size: Number of frames to use for FPS averaging
        """
        self.source = source
        self.width = width
        self.height = height
        self.fps_buffer_size = fps_buffer_size
        
        self.cap = None
        self.frame_times = []
        self.is_running = False
    
    def start(self) -> bool:
        """
        Start the video capture.
        
        Returns:
            bool: True if capture was started successfully, False otherwise
        """
        if self.is_running:
            return True
            
        self.cap = cv2.VideoCapture(self.source)
        
        if not self.cap.isOpened():
            return False
            
        # Try to set properties if it's a webcam
        if isinstance(self.source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        self.is_running = True
        self.frame_times = []
        return True
    
    def stop(self) -> None:
        """Stop the video capture and release resources."""
        if self.is_running and self.cap is not None:
            self.cap.release()
            self.is_running = False
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a single frame from the video source.
        
        Returns:
            Tuple containing:
                - Boolean indicating if frame was successfully read
                - Image as numpy array (or None if no frame was read)
        """
        if not self.is_running or self.cap is None:
            return False, None
            
        # Record time for FPS calculation
        current_time = time.time()
        
        # Read frame
        ret, frame = self.cap.read()
        
        if ret:
            # Update FPS buffer
            self.frame_times.append(current_time)
            if len(self.frame_times) > self.fps_buffer_size:
                self.frame_times.pop(0)
        
        return ret, frame
    
    def get_fps(self) -> float:
        """
        Calculate the current FPS based on actual frame timings.
        
        Returns:
            float: Current frames per second
        """
        if len(self.frame_times) < 2:
            return 0.0
            
        # Calculate FPS from time differences
        time_diff = self.frame_times[-1] - self.frame_times[0]
        if time_diff > 0:
            return (len(self.frame_times) - 1) / time_diff
        return 0.0


def draw_detections(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    font_scale: float = 0.5,
) -> np.ndarray:
    """
    Draw bounding boxes and labels for detected people.
    
    Args:
        image: Input image to draw on
        detections: List of detection results from PeopleDetector
        color: BGR color tuple for bounding boxes
        thickness: Line thickness for bounding boxes
        font_scale: Font scale for text labels
        
    Returns:
        np.ndarray: Image with drawn detections
    """
    annotated_image = image.copy()
    
    for detection in detections:
        # Extract bounding box coordinates
        x_min, y_min, x_max, y_max = detection['box']
        
        # Draw bounding box
        cv2.rectangle(
            annotated_image,
            (x_min, y_min),
            (x_max, y_max),
            color,
            thickness
        )
        
        # Create label with confidence score
        label = f"Person: {detection['score']:.2f}"
        
        # Calculate text size and position
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        
        # Draw label background
        cv2.rectangle(
            annotated_image,
            (x_min, y_min - text_height - 5),
            (x_min + text_width, y_min),
            color,
            -1  # Filled rectangle
        )
        
        # Draw text
        cv2.putText(
            annotated_image,
            label,
            (x_min, y_min - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),  # Black text
            thickness
        )
    
    return annotated_image


def add_performance_stats(
    image: np.ndarray,
    fps: float,
    inference_time: float,
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    text_color: Tuple[int, int, int] = (255, 255, 255),
    font_scale: float = 0.5,
    thickness: int = 1,
) -> np.ndarray:
    """
    Add performance statistics to the image.
    
    Args:
        image: Input image to add stats to
        fps: Current FPS value
        inference_time: Model inference time in seconds
        bg_color: Background color for stats box
        text_color: Text color for stats
        font_scale: Font scale for text
        thickness: Line thickness for text
        
    Returns:
        np.ndarray: Image with added performance stats
    """
    stats_image = image.copy()
    
    # Create stats text
    fps_text = f"FPS: {fps:.1f}"
    inference_text = f"Inference: {inference_time*1000:.1f}ms"
    
    # Get text sizes
    (fps_width, fps_height), _ = cv2.getTextSize(
        fps_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    (inf_width, inf_height), _ = cv2.getTextSize(
        inference_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    
    # Calculate background box dimensions
    box_width = max(fps_width, inf_width) + 20
    box_height = fps_height + inf_height + 20
    
    # Draw background box
    cv2.rectangle(
        stats_image,
        (10, 10),
        (10 + box_width, 10 + box_height),
        bg_color,
        -1  # Filled rectangle
    )
    
    # Draw text
    cv2.putText(
        stats_image,
        fps_text,
        (20, 10 + fps_height + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        text_color,
        thickness
    )
    
    cv2.putText(
        stats_image,
        inference_text,
        (20, 10 + fps_height + inf_height + 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        text_color,
        thickness
    )
    
    return stats_image 