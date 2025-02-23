from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from datetime import datetime
import uuid
from .processor import VideoProcessor
from .models import ProcessingResponse, ProcessingStatus

app = FastAPI(
    title="Fall Detection API",
    description="API for fall detection using YOLOv8 pose estimation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize video processor
video_processor = VideoProcessor()

@app.post("/api/process-video")
async def process_video(video: UploadFile = File(...)):
    """
    Process a video file for fall detection.
    """
    if not video.filename.endswith(('.mp4', '.avi', '.mov')):
        raise HTTPException(400, "Unsupported file format")
    
    # Generate unique ID for this processing
    job_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create temporary file in /tmp (Vercel's writable directory)
    temp_dir = "/tmp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = f"{temp_dir}/{job_id}_{timestamp}.mp4"
    
    try:
        # Save uploaded video
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        
        # Process video
        results = video_processor.process_video(temp_path)
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return ProcessingResponse(**results)
    
    except Exception as e:
        # Clean up on error
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(500, f"Processing failed: {str(e)}")

@app.get("/api/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "healthy"} 