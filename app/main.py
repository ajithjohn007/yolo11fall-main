from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from datetime import datetime
import uuid
import logging
from .processor import VideoProcessor
from .models import ProcessingResponse, ProcessingStatus

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
video_processor = None

@app.on_event("startup")
async def startup_event():
    """Initialize the video processor on startup."""
    global video_processor
    try:
        logger.info("Initializing video processor...")
        video_processor = VideoProcessor()
        logger.info("Video processor initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize video processor: {e}")
        raise

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
    
    # Create temporary directory if it doesn't exist
    temp_dir = "temp/uploads"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Save uploaded video
    temp_path = f"{temp_dir}/{job_id}_{timestamp}.mp4"
    logger.info(f"Saving video to {temp_path}")
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        
        # Process video
        logger.info(f"Processing video {job_id}")
        results = video_processor.process_video(temp_path)
        logger.info(f"Video {job_id} processed successfully")
        
        return ProcessingResponse(**results)
    
    except Exception as e:
        logger.error(f"Processing failed for video {job_id}: {e}")
        raise HTTPException(500, f"Processing failed: {str(e)}")
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
            logger.info(f"Cleaned up temporary file {temp_path}")

@app.get("/api/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {
        "status": "healthy",
        "processor_initialized": video_processor is not None,
        "environment": os.getenv("ENVIRONMENT", "development")
    } 