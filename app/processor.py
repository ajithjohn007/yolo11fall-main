import cv2
import numpy as np
from ultralytics import YOLO
import os
from .utils import process_frame_detections
from typing import Dict, Any

class VideoProcessor:
    def __init__(self, model_path: str = "yolo11s-pose.pt"):
        """Initialize the video processor with the YOLO model."""
        self.model = YOLO(model_path)
    
    def process_video(self, video_path: str) -> Dict[str, Any]:
        """
        Process a video file for fall detection.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary containing processing results
        """
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "Failed to open video file"}
        
        # Get video properties
        width = int(1020)
        height = int(600)
        fps = int(cap.get(cv2.CAP_PROP_FPS) // 2)
        
        detections = []
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                if frame_count % 3 != 0:  # Skip frames for performance
                    continue
                    
                frame = cv2.resize(frame, (width, height))
                
                # Run YOLOv8 tracking with pose estimation
                results = self.model.track(frame, persist=True, classes=0)
                
                # Process detections
                frame_detections = process_frame_detections(results, frame_count, fps)
                detections.extend(frame_detections)
        
        finally:
            cap.release()
        
        return {
            "total_frames": frame_count,
            "fps": fps,
            "detections": detections
        }

    def cleanup(self, video_path: str):
        """Clean up temporary files after processing."""
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
        except Exception as e:
            print(f"Error cleaning up {video_path}: {e}") 