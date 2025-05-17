"""
Performance testing for people detection models.

This script tests each model (YOLOv8n, YOLOv8s, YOLOv8m, and custom models)
with CPU limitations to measure latency, FPS, and resource consumption.
"""

import os
import time
import json
import psutil
import csv
from pathlib import Path
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# Fix import path
import sys
sys.path.append('/app/real-time-people-detection')
from app.detection import PeopleDetector

# Set CPU limit if possible (this will be handled by Docker)
# We'll still track CPU usage for reporting

# Define test parameters
NUM_TEST_FRAMES = 100  # Number of frames to process for each test
REPEAT_TESTS = 3  # Number of times to repeat each test for reliability

# Define models to test with correct paths
MODELS = {
    "YOLOv8n": "/app/real-time-people-detection/yolov8n.pt",
    "YOLOv8s": "/app/real-time-people-detection/yolov8s.pt", 
    "YOLOv8m": "/app/real-time-people-detection/yolov8m.pt",
    # Add custom models - fixed paths for Docker
    "YOLO11n": "/app/real-time-people-detection/yolo11n_results/best.pt",
    "YOLO11s": "/app/real-time-people-detection/yolo11s_results/best.pt",
    "YOLO11m": "/app/real-time-people-detection/yolo11m_results/best.pt",
}

# Define test videos with correct paths
TEST_VIDEOS = {
    "one_person": "/app/real-time-people-detection/assets/one-by-one-person-detection.mp4",
    "store_aisle": "/app/real-time-people-detection/assets/store-aisle-detection.mp4",
    "people_detection": "/app/real-time-people-detection/assets/people-detection.mp4",
}

# Define result storage directory
RESULTS_DIR = Path("/app/performance_results")
RESULTS_DIR.mkdir(exist_ok=True)


def get_memory_usage():
    """Get current memory usage of the process in MB."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / (1024 * 1024)  # Convert to MB


def get_cpu_usage():
    """Get current CPU usage as percentage."""
    return psutil.cpu_percent(interval=0.1)


def test_model(model_name, model_path, video_path, threshold=0.5):
    """
    Test a single model on a video and return performance metrics.
    
    Args:
        model_name: Name of the model
        model_path: Path to the model file
        video_path: Path to the test video
        threshold: Detection confidence threshold
        
    Returns:
        dict: Performance metrics
    """
    print(f"Testing {model_name} on {Path(video_path).name}...")
    
    # Initialize metrics storage
    latencies = []
    cpu_usages = []
    memory_usages = []
    inference_times = []
    detection_counts = []
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video at {video_path}")
    
    # Get total frame count for progress tracking
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_process = min(NUM_TEST_FRAMES, total_frames)
    
    # Initialize detector
    initial_memory = get_memory_usage()
    detector = PeopleDetector(model_name=model_path, threshold=threshold)
    model_size_memory = get_memory_usage() - initial_memory
    
    # Process video frames
    for test_iteration in range(REPEAT_TESTS):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning of video
        
        for frame_idx in tqdm(range(frames_to_process), desc=f"Test {test_iteration+1}/{REPEAT_TESTS}"):
            # Measure frame processing time
            start_time = time.time()
            
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
                
            # Record CPU and memory before inference
            pre_cpu = get_cpu_usage()
            pre_memory = get_memory_usage()
            
            # Perform detection
            detections, inference_time = detector.detect(frame)
            
            # Record metrics
            end_time = time.time()
            latency = end_time - start_time
            
            # Record CPU and memory after inference
            post_cpu = get_cpu_usage()
            post_memory = get_memory_usage()
            
            # Store metrics
            latencies.append(latency)
            inference_times.append(inference_time)
            cpu_usages.append(post_cpu - pre_cpu if post_cpu > pre_cpu else post_cpu)
            memory_usages.append(post_memory)
            detection_counts.append(len(detections))
    
    # Release resources
    cap.release()
    
    # Calculate metrics
    avg_latency = np.mean(latencies)
    avg_fps = 1.0 / avg_latency if avg_latency > 0 else 0
    avg_inference_time = np.mean(inference_times)
    avg_cpu_usage = np.mean(cpu_usages)
    avg_memory_usage = np.mean(memory_usages)
    avg_detection_count = np.mean(detection_counts)
    
    # Return metrics
    return {
        "model": model_name,
        "video": Path(video_path).name,
        "avg_latency_ms": avg_latency * 1000,  # Convert to ms
        "avg_fps": avg_fps,
        "avg_inference_time_ms": avg_inference_time * 1000,  # Convert to ms
        "avg_cpu_usage_percent": avg_cpu_usage,
        "avg_memory_usage_mb": avg_memory_usage,
        "peak_memory_usage_mb": max(memory_usages),
        "model_size_memory_mb": model_size_memory,
        "avg_detections_per_frame": avg_detection_count
    }


def run_all_tests():
    """Run performance tests for all models on all videos."""
    all_results = []
    
    for model_name, model_path in MODELS.items():
        for video_name, video_path in TEST_VIDEOS.items():
            try:
                # Run test
                results = test_model(model_name, model_path, video_path)
                all_results.append(results)
                
                # Print results
                print(f"\nResults for {model_name} on {video_name}:")
                print(f"  Average Latency: {results['avg_latency_ms']:.2f} ms")
                print(f"  Average FPS: {results['avg_fps']:.2f}")
                print(f"  Average Inference Time: {results['avg_inference_time_ms']:.2f} ms")
                print(f"  Average CPU Usage: {results['avg_cpu_usage_percent']:.2f}%")
                print(f"  Average Memory Usage: {results['avg_memory_usage_mb']:.2f} MB")
                print(f"  Average Detections: {results['avg_detections_per_frame']:.2f} per frame")
                print("\n" + "-"*50)
                
            except Exception as e:
                print(f"Error testing {model_name} on {video_name}: {e}")
    
    return all_results


def save_results(results):
    """Save test results to CSV and JSON files."""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Save as CSV
    csv_path = RESULTS_DIR / f"performance_results_{timestamp}.csv"
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = results[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    # Save as JSON
    json_path = RESULTS_DIR / f"performance_results_{timestamp}.json"
    with open(json_path, 'w') as jsonfile:
        json.dump(results, jsonfile, indent=2)
    
    print(f"Results saved to {csv_path} and {json_path}")
    
    return csv_path, json_path


def generate_charts(results, output_dir=RESULTS_DIR):
    """Generate performance comparison charts."""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Group results by model
    models = list(set(r["model"] for r in results))
    videos = list(set(r["video"] for r in results))
    
    # Create performance comparison charts
    metrics = [
        {"name": "avg_fps", "title": "Average FPS", "higher_better": True},
        {"name": "avg_latency_ms", "title": "Average Latency (ms)", "higher_better": False},
        {"name": "avg_cpu_usage_percent", "title": "CPU Usage (%)", "higher_better": False},
        {"name": "avg_memory_usage_mb", "title": "Memory Usage (MB)", "higher_better": False},
    ]
    
    for metric in metrics:
        plt.figure(figsize=(12, 8))
        
        # Group by video type
        for video in videos:
            video_results = [r for r in results if r["video"] == video]
            video_results.sort(key=lambda x: x[metric["name"]], reverse=metric["higher_better"])
            
            model_names = [r["model"] for r in video_results]
            metric_values = [r[metric["name"]] for r in video_results]
            
            x = range(len(model_names))
            plt.bar([i + videos.index(video) * 0.25 for i in x], metric_values, width=0.2, 
                   label=f"{video.replace('.mp4', '')}")
        
        plt.xlabel("Model")
        plt.ylabel(metric["title"])
        plt.title(f"Model Performance Comparison - {metric['title']}")
        plt.xticks(range(len(models)), models, rotation=45)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save figure
        chart_path = output_dir / f"{metric['name']}_comparison_{timestamp}.png"
        plt.savefig(chart_path)
        print(f"Saved chart to {chart_path}")
    
    # Memory usage vs FPS chart (bubble chart)
    plt.figure(figsize=(10, 8))
    
    for model in models:
        model_results = [r for r in results if r["model"] == model]
        
        for r in model_results:
            plt.scatter(r["avg_memory_usage_mb"], r["avg_fps"], 
                       s=r["avg_cpu_usage_percent"] * 10, alpha=0.7, label=f"{model}_{r['video']}")
    
    plt.xlabel("Memory Usage (MB)")
    plt.ylabel("FPS")
    plt.title("Performance Trade-off: Memory vs FPS")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    tradeoff_path = output_dir / f"memory_fps_tradeoff_{timestamp}.png"
    plt.savefig(tradeoff_path)
    print(f"Saved trade-off chart to {tradeoff_path}")


def main():
    """Main test function."""
    print("Starting performance tests...")
    print(f"Testing {len(MODELS)} models on {len(TEST_VIDEOS)} videos")
    print(f"Each test processes {NUM_TEST_FRAMES} frames and is repeated {REPEAT_TESTS} times")
    
    # Run all tests
    results = run_all_tests()
    
    # Save results
    csv_path, json_path = save_results(results)
    
    # Generate charts
    generate_charts(results)
    
    print("\nPerformance testing complete!")
    print(f"Results saved to {csv_path}")
    

if __name__ == "__main__":
    main() 