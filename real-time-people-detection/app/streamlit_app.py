"""
Streamlit application for real-time people detection.

This module provides a web interface for the people detection system,
allowing users to configure settings and view the detection results in real-time.
"""

import os
import time
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from app.detection import PeopleDetector
from app.utils import VideoSource, draw_detections, add_performance_stats


# Constants
ASSETS_DIR = Path(__file__).parent.parent / "assets"
# Add custom model paths
MODELS_DIR = Path(__file__).parent.parent.parent
YOLO11N_MODEL = MODELS_DIR / "yolo11n_results" / "best.pt"
YOLO11S_MODEL = MODELS_DIR / "yolo11s_results" / "best.pt"
YOLO11M_MODEL = MODELS_DIR / "yolo11m_results" / "best.pt"

DEMO_VIDEOS = {
    "One Person": ASSETS_DIR / "one-by-one-person-detection.mp4",
    "Store Aisle": ASSETS_DIR / "store-aisle-detection.mp4",
    "People Detection": ASSETS_DIR / "people-detection.mp4"
}
FRAME_WIDTH = 640
FRAME_HEIGHT = 480


class PeopleDetectionApp:
    """
    Streamlit application for real-time people detection.
    
    This class handles the Streamlit UI components and orchestrates
    the video capture and detection processes.
    """
    
    def __init__(self):
        """Initialize the Streamlit application components."""
        # Set page config
        st.set_page_config(
            page_title="Real-time People Detection",
            page_icon="👁️",
            layout="wide",
        )
        
        # Initialize session state
        if "video_source" not in st.session_state:
            st.session_state.video_source = None
        if "detector" not in st.session_state:
            st.session_state.detector = None
        if "is_running" not in st.session_state:
            st.session_state.is_running = False
        if "frame_placeholder" not in st.session_state:
            st.session_state.frame_placeholder = None
        if "last_inference_time" not in st.session_state:
            st.session_state.last_inference_time = 0.0
        if "last_inference_timestamp" not in st.session_state:
            st.session_state.last_inference_timestamp = 0.0
        if "frame_count" not in st.session_state:
            st.session_state.frame_count = 0
        if "last_frame" not in st.session_state:
            st.session_state.last_frame = None
        if "last_detections" not in st.session_state:
            st.session_state.last_detections = []
            
    def create_ui(self):
        """Create the Streamlit UI components."""
        # Page header
        st.title("Real-time People Detection")
        st.markdown(
            "This application detects people in video streams using YOLOv8."
        )
        
        # Sidebar for controls
        with st.sidebar:
            st.header("Settings")
            
            # Create a mapping for readable model names
            model_options = {
                "YOLOv8n (Nano)": "yolov8n.pt",
                "YOLOv8s (Small)": "yolov8s.pt",
                "YOLOv8m (Medium)": "yolov8m.pt",
                "YOLO11n (Custom)": str(YOLO11N_MODEL),
                "YOLO11s (Custom)": str(YOLO11S_MODEL),
                "YOLO11m (Custom)": str(YOLO11M_MODEL),
            }
            
            # Model selection
            model_display_name = st.selectbox(
                "Select detection model",
                options=list(model_options.keys()),
                index=0,
            )
            
            # Get the actual model path from the display name
            model_name = model_options[model_display_name]
            
            # Detection threshold
            detection_threshold = st.slider(
                "Detection threshold",
                min_value=0.1,
                max_value=1.0,
                value=0.5,
                step=0.05,
            )

            # Target inference FPS
            target_fps = st.slider(
                "Target inference FPS",
                min_value=1,
                max_value=30,
                value=10,
                step=1,
                help="Control how many frames per second are sent to the model for inference. Lower values use less resources but may appear less smooth."
            )
            
            # Video selection
            demo_selection = st.selectbox(
                "Select demo video",
                options=list(DEMO_VIDEOS.keys()),
                index=0,
            )
            video_path = str(DEMO_VIDEOS[demo_selection])
            
            # Control buttons
            col1, col2 = st.columns(2)
            
            with col1:
                start_button = st.button(
                    "Start" if not st.session_state.is_running else "Restart",
                    use_container_width=True,
                )
            
            with col2:
                stop_button = st.button(
                    "Stop",
                    use_container_width=True,
                    disabled=not st.session_state.is_running,
                )
        
        # Main area for video display
        video_column, stats_column = st.columns([3, 1])
        
        with video_column:
            st.subheader("Detection Feed")
            # Create a placeholder for the video frame
            frame_placeholder = st.empty()
            st.session_state.frame_placeholder = frame_placeholder
        
        with stats_column:
            st.subheader("Performance Stats")
            # Create placeholders for stats
            fps_text = st.empty()
            inference_text = st.empty()
            people_count = st.empty()
            inference_fps_text = st.empty()
        
        # Handle button actions
        if start_button:
            self.start_detection(video_path, model_name, detection_threshold, target_fps)
        
        if stop_button:
            self.stop_detection()
            
        # Return stats placeholders for updating
        return fps_text, inference_text, people_count, inference_fps_text
    
    def start_detection(self, source, model_name, threshold, target_fps):
        """
        Start the detection process.
        
        Args:
            source: Video file path
            model_name: YOLOv8 model to use
            threshold: Detection confidence threshold
            target_fps: Target frames per second for inference
        """
        # Stop existing detection if running
        self.stop_detection()
        
        # Initialize detector
        detector = PeopleDetector(
            model_name=model_name,
            threshold=threshold,
        )
        
        # Initialize VideoSource
        video_source = VideoSource(
            source=source,
            width=FRAME_WIDTH,
            height=FRAME_HEIGHT,
        )
        if not video_source.start():
            st.error(f"Failed to open video source: {source}")
            return
        
        # Store detector and state
        st.session_state.video_source = video_source
        st.session_state.detector = detector
        st.session_state.is_running = True
        st.session_state.target_fps = target_fps
        st.session_state.last_inference_timestamp = time.time()
        st.session_state.frame_count = 0
        st.session_state.last_frame = None
        st.session_state.last_detections = []
    
    def stop_detection(self):
        """Stop the detection process and release resources."""
        if st.session_state.video_source is not None:
            st.session_state.video_source.stop()
            st.session_state.video_source = None
        
        st.session_state.detector = None
        st.session_state.is_running = False
        st.session_state.last_frame = None
        st.session_state.last_detections = []
    
    def update_frame(self, fps_text, inference_text, people_count, inference_fps_text):
        """
        Update the video frame and stats.
        
        Args:
            fps_text: Streamlit element for FPS display
            inference_text: Streamlit element for inference time display
            people_count: Streamlit element for people count display
            inference_fps_text: Streamlit element for inference FPS display
        """
        if not st.session_state.is_running:
            return
        
        video_source = st.session_state.video_source
        detector = st.session_state.detector
        target_fps = st.session_state.target_fps
        
        if video_source is None or detector is None:
            return
        
        # Read frame from video source
        ret, frame = video_source.read_frame()
        
        if not ret or frame is None:
            # Video has ended
            st.warning("Video has ended. Choose another video or restart.")
            self.stop_detection()
            return False
        
        # Store the latest frame even if we don't run inference on it
        st.session_state.last_frame = frame
        st.session_state.frame_count += 1
        
        # Calculate time since last inference
        current_time = time.time()
        time_since_last_inference = current_time - st.session_state.last_inference_timestamp
        
        # Determine if we should run inference based on target FPS
        should_run_inference = time_since_last_inference >= (1.0 / target_fps)
        
        # Track actual inference FPS
        inference_fps = 1.0 / time_since_last_inference if time_since_last_inference > 0 and should_run_inference else 0
        
        if should_run_inference:
            # Run detection
            detections, inference_time = detector.detect(frame)
            
            # Update inference timestamp
            st.session_state.last_inference_timestamp = current_time
            
            # Update last inference time and detections
            st.session_state.last_inference_time = inference_time
            st.session_state.last_detections = detections
        else:
            # Use the last frame's detections
            detections = st.session_state.last_detections
            inference_time = st.session_state.last_inference_time
        
        # Draw detections on frame
        annotated_frame = draw_detections(frame, detections)
        
        # Add performance stats
        annotated_frame = add_performance_stats(
            annotated_frame,
            video_source.get_fps(),
            inference_time,
        )
        
        # Convert to RGB for Streamlit display
        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        # Display the frame
        st.session_state.frame_placeholder.image(
            rgb_frame,
            caption="Detection Results",
            use_container_width=True,
        )
        
        # Update stats
        actual_fps = video_source.get_fps()
        fps_text.markdown(f"**FPS:** {actual_fps:.1f}")
        inference_text.markdown(f"**Inference Time:** {inference_time*1000:.1f} ms")
        people_count.markdown(f"**People Detected:** {len(detections)}")
        
        # Display actual inference rate
        if should_run_inference:
            inference_fps_text.markdown(f"**Inference Rate:** {inference_fps:.1f} FPS (Target: {target_fps})")
        else:
            inference_fps_text.markdown(f"**Inference Rate:** {target_fps} FPS (Target: {target_fps})")
        
        return True


def main():
    """Main function to run the Streamlit application."""
    # Create the detection app
    app = PeopleDetectionApp()
    
    # Create UI
    fps_text, inference_text, people_count, inference_fps_text = app.create_ui()
    
    # Run update loop while the app is active
    if st.session_state.is_running:
        while app.update_frame(fps_text, inference_text, people_count, inference_fps_text):
            # Add a small delay to prevent excessive CPU usage
            time.sleep(0.01)
    

if __name__ == "__main__":
    main() 