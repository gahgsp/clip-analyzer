from pathlib import Path
from fastapi import APIRouter, HTTPException

from app.model.clip import ClipRequest, ClipResponse, FrameAnalysis
from app.service import clip_service, analysis_service

FRAMES_DIR = Path("static/frames")
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

router = APIRouter()
clip_service = clip_service.ClipService()
analysis_service = analysis_service.AnalysisService()


@router.post(path="/", response_model=ClipResponse)
def process(request: ClipRequest):
    try:
        clip_data = clip_service.process_clip(url=request.url)

        frame_paths = clip_data["frames"]
        analyzed_frames = []

        for frame_path in frame_paths:
            image_description = analysis_service.analyze_frame(frame_path)
            analyzed_frames.append(FrameAnalysis(
                path=frame_path, description=image_description))

        return ClipResponse(clip_id=clip_data["clip_id"], duration=clip_data["duration"], frames=analyzed_frames)
    except HTTPException as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
