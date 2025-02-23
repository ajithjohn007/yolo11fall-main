from pydantic import BaseModel
from typing import List, Tuple, Optional

class Detection(BaseModel):
    frame_number: int
    time: float
    track_id: int
    status: str
    confidence: float
    fall_score: float
    bounding_box: Tuple[int, int, int, int]

class ProcessingResponse(BaseModel):
    total_frames: int
    fps: int
    detections: List[Detection]

class ProcessingStatus(BaseModel):
    job_id: str
    status: str
    message: Optional[str] = None 