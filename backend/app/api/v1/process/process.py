from pathlib import Path
from fastapi import APIRouter, HTTPException

from app.model.clip import ClipRequest, ClipResponse
from app.service import clip_service

FRAMES_DIR = Path("static/frames")
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

router = APIRouter()
clip_service = clip_service.ClipService()


class ProcessService:
    @router.post(path="/", response_model=ClipResponse)
    def process(request: ClipRequest):
        try:
            return clip_service.process_clip(url=request.url)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except IOError as e:
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            # Our fallback in case we got an unexpected error when processing the clip.
            raise HTTPException(
                status_code=500, detail=f"Unexpected error: {str(e)}.")
