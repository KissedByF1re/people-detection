"""
Detection module for identifying people in video frames.

This module uses a pre-trained YOLOv8n model from Ultralytics to detect people in images.
It handles both CPU and GPU acceleration and provides configurability for thresholds.
"""

import time
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import torch
from ultralytics import YOLO


class PeopleDetector:
    """
    A class for detecting people in images using a pre-trained YOLOv8n model.
    
    Attributes:
        model_name: Name or path of the YOLOv8 model to use
        threshold: Confidence threshold for detection
        device: Device to run inference on (cuda/cpu)
        model: The detection model
    """
    
    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        threshold: float = 0.5,
        device: Optional[str] = None,
    ):
        """
        Initialize the people detector with a pre-trained model.
        
        Args:
            model_name: YOLOv8 model name to use ('yolov8n.pt' is the smallest one)
            threshold: Confidence threshold for detection (0.0 to 1.0)
            device: Device to run inference on (cuda/cpu). If None, will use cuda if available.
        """
        self.model_name = model_name
        self.threshold = threshold
        
        # Determine the device to use
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Load the YOLOv8 model
        self.model = YOLO(model_name)
        
        # Person class ID is 0 in COCO (YOLOv8 uses COCO classes)
        self.person_class_id = 0

    def detect(self, image: np.ndarray) -> Tuple[List[Dict[str, Any]], float]:
        """
        Detect people in an image.
        
        Args:
            image: Input image as numpy array (BGR format from OpenCV)
            
        Returns:
            Tuple containing:
                - List of detection results with keys 'box', 'score', and 'label'
                - Inference time in seconds
        """
        # Start timing
        start_time = time.time()
        
        # Run inference with YOLOv8
        results = self.model(image, conf=self.threshold, device=self.device)
        
        # Extract detections of people only
        detections = []
        
        # Process the results
        for result in results:
            boxes = result.boxes
            
            # Extract coordinates, confidence and class
            for i, box in enumerate(boxes):
                cls = int(box.cls.item())
                conf = float(box.conf.item())
                
                # Check if it's a person (class 0)
                if cls == self.person_class_id:
                    # Get bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy.tolist()[0])
                    
                    detections.append({
                        'box': (x1, y1, x2, y2),
                        'score': conf,
                        'label': 'person'
                    })
        
        # Calculate inference time
        inference_time = time.time() - start_time
        
        return detections, inference_time
    
    def update_threshold(self, threshold: float) -> None:
        """
        Update the detection confidence threshold.
        
        Args:
            threshold: New threshold value (0.0 to 1.0)
        """
        self.threshold = threshold 