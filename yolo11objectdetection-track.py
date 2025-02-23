import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import os
from datetime import datetime
from collections import deque

# Create results directory if it doesn't exist
if not os.path.exists('results'):
    os.makedirs('results')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

def calculate_fall_metrics(keypoints):
    """Calculate various metrics to detect falls using pose keypoints"""
    if keypoints is None or len(keypoints) == 0:
        return False, 0
    
    # Extract relevant keypoints (assuming COCO format)
    try:
        nose = keypoints[0]
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        
        # Calculate center points
        shoulder_center = [(left_shoulder[0] + right_shoulder[0])/2, 
                         (left_shoulder[1] + right_shoulder[1])/2]
        hip_center = [(left_hip[0] + right_hip[0])/2, 
                     (left_hip[1] + right_hip[1])/2]
        
        # Calculate angles and distances
        vertical_angle = np.abs(np.arctan2(shoulder_center[0] - hip_center[0],
                                         shoulder_center[1] - hip_center[1]))
        vertical_angle_degrees = np.degrees(vertical_angle)
        
        # Height to width ratio of the body
        body_height = np.sqrt((shoulder_center[0] - hip_center[0])**2 + 
                            (shoulder_center[1] - hip_center[1])**2)
        shoulder_width = np.sqrt((left_shoulder[0] - right_shoulder[0])**2 + 
                               (left_shoulder[1] - right_shoulder[1])**2)
        height_width_ratio = body_height / (shoulder_width + 1e-6)
        
        # Calculate head position relative to hips
        head_hip_distance = np.sqrt((nose[0] - hip_center[0])**2 + 
                                  (nose[1] - hip_center[1])**2)
        normal_head_hip_distance = head_hip_distance / body_height
        
        # Combined fall score based on multiple metrics
        fall_score = 0
        
        if vertical_angle_degrees < 45 or vertical_angle_degrees > 135:
            fall_score += 0.4
        if height_width_ratio < 1.2:
            fall_score += 0.3
        if normal_head_hip_distance < 0.5:
            fall_score += 0.3
            
        return fall_score > 0.5, fall_score
        
    except Exception as e:
        print(f"Error in fall detection calculation: {e}")
        return False, 0

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Load the YOLOv8 model
model = YOLO("yolo11s-pose.pt")
names = model.model.names

# Open the video file
cap = cv2.VideoCapture('fall.mp4')

# Get video properties
width = int(1020)
height = int(600)
fps = int(cap.get(cv2.CAP_PROP_FPS) // 2)  # Halve the FPS

# Buffer settings
BUFFER_SECONDS = 3  # Seconds to keep before and after fall
buffer_size = BUFFER_SECONDS * fps
frame_buffer = deque(maxlen=buffer_size)
recording = False
post_fall_frames = 0
current_video_writer = None
fall_count = 0

# Initialize log file
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_path = os.path.join('results', f'detection_log_{timestamp}.txt')

with open(log_path, 'w') as log_file:
    log_file.write("Time,TrackID,Status,Confidence,FallScore,BoundingBox,VideoFile\n")
    
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        count += 1
        if count % 3 != 0:  # Skip frames for performance
            continue
            
        frame = cv2.resize(frame, (width, height))
        frame_buffer.append(frame.copy())
        
        # Run YOLOv8 tracking with pose estimation
        results = model.track(frame, persist=True, classes=0)
        
        fall_detected = False
        current_frame_data = []  # Store frame's detection data
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            confidences = results[0].boxes.conf.cpu().tolist()
            keypoints = results[0].keypoints.data.cpu().numpy() if results[0].keypoints else None
            
            for i, (box, class_id, track_id, conf) in enumerate(zip(boxes, class_ids, track_ids, confidences)):
                x1, y1, x2, y2 = box
                
                # Get keypoints for this detection
                person_keypoints = keypoints[i] if keypoints is not None else None
                is_falling, fall_score = calculate_fall_metrics(person_keypoints)
                
                # Additional traditional metrics
                h = y2-y1
                w = x2-x1
                aspect_ratio = h/w if w > 0 else float('inf')
                
                frame_time = count/fps
                
                # Determine fall status
                if is_falling or aspect_ratio < 1.2:
                    status = "Fall"
                    color = (0, 0, 255)
                    fall_detected = True
                    cv2.circle(frame, (x1, y1), 5, (0, 0, 255), -1)
                    cvzone.putTextRect(frame, f"FALL DETECTED!", (x1, y1-25), 2, 2, colorR=(255,0,0))
                else:
                    status = "Normal"
                    color = (0, 255, 0)
                
                # Draw bounding box and labels
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                cvzone.putTextRect(frame, f'ID: {track_id}', (x1,y2), 1, 1)
                cvzone.putTextRect(frame, f'{status}', (x1,y1), 1, 1)
                cvzone.putTextRect(frame, f'Fall Score: {fall_score:.2f}', (x1,y2+25), 1, 1)
                
                # Store detection data
                current_frame_data.append({
                    'time': frame_time,
                    'track_id': track_id,
                    'status': status,
                    'conf': conf,
                    'fall_score': fall_score,
                    'box': (x1,y1,x2,y2)
                })
        
        # Handle recording logic
        if fall_detected and not recording:
            # Start new recording
            fall_count += 1
            video_name = f'fall_detection_{timestamp}_incident_{fall_count}.mp4'
            output_path = os.path.join('results', video_name)
            current_video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
            
            # Write buffered frames
            for buffered_frame in frame_buffer:
                current_video_writer.write(buffered_frame)
            
            recording = True
            post_fall_frames = buffer_size  # Reset post-fall counter
            
            # Log fall incident
            for data in current_frame_data:
                log_entry = f"{data['time']:.2f},{data['track_id']},{data['status']},{data['conf']:.2f},"
                log_entry += f"{data['fall_score']:.2f},{data['box']},{video_name}\n"
                log_file.write(log_entry)
        
        elif recording:
            if fall_detected:
                post_fall_frames = buffer_size  # Reset counter if fall is still occurring
            else:
                post_fall_frames -= 1
            
            # Write current frame
            current_video_writer.write(frame)
            
            # Check if we should stop recording
            if post_fall_frames <= 0:
                current_video_writer.release()
                current_video_writer = None
                recording = False
        
        cv2.imshow("RGB", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Release resources
if current_video_writer is not None:
    current_video_writer.release()
cap.release()
cv2.destroyAllWindows()

print(f"Results saved to results folder:")
print(f"- Log: {log_path}")
print(f"- Fall detection videos saved as separate files")