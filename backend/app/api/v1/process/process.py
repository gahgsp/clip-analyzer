from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException

from app.model.clip import ClipRequest, ClipResponse, FrameAnalysis
from app.core.exceptions import ClipTooLongError, StreamResolutionError
from app.service.clip_service import ClipService
from app.core.dependencies import get_analysis_service, get_clip_service, get_langchain_service
from app.service.analysis_service import AnalysisService
from app.service.lgc_service import LangChainService

FRAMES_DIR = Path("static/frames")
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

router = APIRouter()


@router.post(path="/", response_model=ClipResponse)
def process(request: ClipRequest,
            clip_service: ClipService = Depends(get_clip_service),
            analysis_service: AnalysisService = Depends(get_analysis_service)):
    try:
        clip_data = clip_service.process_clip(url=request.url)

        frame_paths = clip_data.frame_paths
        analyzed_frames = []

        analysises = analysis_service.analyze_frames(frame_paths)

        for index, frame_path in enumerate(frame_paths):
            analyzed_frames.append(FrameAnalysis(
                path=frame_path, description=analysises[index]))

        summary = analysis_service.generate_summary(
            descriptions=[analyzed_frame.description for analyzed_frame in analyzed_frames])

        return ClipResponse(clip_id=clip_data.clip_id, duration=clip_data.duration, frames=analyzed_frames, summary=summary)
    except ClipTooLongError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except StreamResolutionError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An unexpected server error happened: {str(e)}.")


@router.post(path="/langchain", response_model=ClipResponse)
def langchain_process(
    request: ClipRequest,
    pipeline: LangChainService = Depends(
        get_langchain_service),
) -> ClipResponse:
    try:
        # Here we call the entire chain that we built.
        result = pipeline.chain.invoke(request.url)

        return ClipResponse(
            clip_id=result.clip_id,
            duration=result.duration,
            frames=result.frames,
            summary=result.summary,
        )
    except ClipTooLongError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except StreamResolutionError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An unexpected server error happened: {str(e)}.")
