import numpy as np
from typing import List, Tuple

def calculate_fall_metrics(keypoints: np.ndarray) -> Tuple[float, bool]:
    """
    Calculate fall metrics based on pose keypoints.
    
    Args:
        keypoints: Numpy array of pose keypoints [N, 3]
    
    Returns:
        Tuple of (fall_score, is_fall)
    """
    if keypoints is None or len(keypoints) == 0:
        return 0.0, False

    # Get relevant keypoints for fall detection
    # Using COCO keypoint format
    shoulders = keypoints[[5, 6]]  # Left and right shoulders
    hips = keypoints[[11, 12]]    # Left and right hips
    ankles = keypoints[[15, 16]]  # Left and right ankles

    # Calculate vertical orientation
    torso_vector = np.mean(shoulders, axis=0)[:2] - np.mean(hips, axis=0)[:2]
    vertical_angle = np.abs(np.arctan2(torso_vector[1], torso_vector[0]))
    
    # Calculate body spread
    points = np.vstack([shoulders, hips, ankles])
    valid_points = points[points[:, 2] > 0.2]  # Filter low confidence points
    if len(valid_points) < 3:
        return 0.0, False
    
    spread = np.max(np.std(valid_points[:, :2], axis=0))
    
    # Combine metrics
    fall_score = (1 - abs(vertical_angle - np.pi/2) / (np.pi/2)) * 0.7 + (spread / 100) * 0.3
    is_fall = fall_score > 0.6

    return float(fall_score), is_fall

def process_frame_detections(results, frame_number: int, fps: float) -> List[dict]:
    """
    Process YOLO detections for a single frame.
    
    Args:
        results: YOLO results object
        frame_number: Current frame number
        fps: Video FPS
    
    Returns:
        List of detection dictionaries
    """
    detections = []
    time = frame_number / fps

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes
        poses = results[0].keypoints.data

        for i, (box, pose) in enumerate(zip(boxes, poses)):
            track_id = int(box.id.item())
            confidence = float(box.conf.item())
            
            # Calculate fall metrics
            fall_score, is_fall = calculate_fall_metrics(pose.cpu().numpy())
            
            # Get bounding box
            x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
            
            detection = {
                "frame_number": frame_number,
                "time": time,
                "track_id": track_id,
                "status": "fall" if is_fall else "normal",
                "confidence": confidence,
                "fall_score": fall_score,
                "bounding_box": (x1, y1, x2, y2)
            }
            detections.append(detection)
    
    return detections 