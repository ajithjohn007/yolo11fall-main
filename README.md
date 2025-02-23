# Fall Detection API

A FastAPI application for fall detection using YOLOv8 pose estimation, deployed on Vercel.

## Deployment to Vercel

1. Install Vercel CLI:
```bash
npm i -g vercel
```

2. Login to Vercel:
```bash
vercel login
```

3. Deploy the application:
```bash
vercel
```

## API Endpoints

### Process Video
```http
POST /api/process-video
```
Upload and process a video file for fall detection.

**Request Body:**
- `video`: Video file (mp4, avi, or mov)

**Response:**
```json
{
    "total_frames": 100,
    "fps": 30,
    "detections": [
        {
            "frame_number": 1,
            "time": 0.033,
            "track_id": 1,
            "status": "normal",
            "confidence": 0.95,
            "fall_score": 0.1,
            "bounding_box": [100, 100, 200, 200]
        }
    ]
}
```

### Health Check
```http
GET /api/health
```
Check if the API is running.

## Important Notes

1. This application is designed for Vercel's serverless environment
2. Video processing is done synchronously due to serverless constraints
3. Temporary files are stored in `/tmp` directory
4. Maximum video file size is limited by Vercel's payload limits (100MB)
5. Processing time is limited by Vercel's function timeout (10s for Hobby, 60s for Pro)

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000` 