from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from datetime import datetime
import uuid
import sys
import logging

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.processor import VideoProcessor
from app.models import ProcessingResponse, ProcessingStatus

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize video processor
video_processor = None

@app.on_event("startup")
async def startup_event():
    global video_processor
    try:
        video_processor = VideoProcessor()
    except Exception as e:
        logging.error(f"Failed to initialize video processor: {e}")

@app.post("/api/process-video")
async def process_video(video: UploadFile = File(...)):
    """
    Process a video file for fall detection.
    """
    if not video.filename.endswith(('.mp4', '.avi', '.mov')):
        raise HTTPException(400, "Unsupported file format")
    
    if video_processor is None:
        raise HTTPException(500, "Video processor not initialized")
    
    # Generate unique ID for this processing
    job_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create temporary file in /tmp
    temp_dir = "/tmp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = f"{temp_dir}/{job_id}_{timestamp}.mp4"
    
    try:
        # Save uploaded video
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        
        # Process video
        results = video_processor.process_video(temp_path)
        
        return ProcessingResponse(**results)
    
    except Exception as e:
        logging.error(f"Processing failed: {e}")
        raise HTTPException(500, f"Processing failed: {str(e)}")
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/api/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "healthy", "processor_initialized": video_processor is not None} 