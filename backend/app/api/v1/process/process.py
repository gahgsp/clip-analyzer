from pathlib import Path
import cv2
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import yt_dlp

FRAMES_DIR = Path("static/frames")
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

router = APIRouter()


class ClipRequest(BaseModel):
    url: str


@router.post("/")
async def process(request: ClipRequest):
    try:
        # TODO: I had to add the "nocheckcertificate" for it to work. Is this the best way?
        with yt_dlp.YoutubeDL(params={"quiet": True, "noplaylist": True, "nocheckcertificate": True}) as ydl:
            info = ydl.extract_info(url=request.url, download=False)
            stream_url = info.get("url", "")
            duration = info.get("duration", 0)
            clip_id = info.get("id", "unknown")
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to resolve the Twitch URL: {str(e)}")

    captured_frames = []
    capture = cv2.VideoCapture(filename=stream_url)

    if not capture.isOpened():
        raise HTTPException(
            status_code=500, detail=f"Could not open the video stream from: {str(stream_url)}")

    # TODO: Make this be adjustable through the API.
    percentages = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

    for percentage in percentages:
        timestamp_in_ms = (duration * percentage) * 1000
        capture.set(propId=cv2.CAP_PROP_POS_MSEC, value=timestamp_in_ms)
        success, frame = capture.read()

        if success:
            filename = f"{clip_id}_{int(percentage*100)}.jpg"
            filepath = FRAMES_DIR / filename
            cv2.imwrite(filename=str(filepath), img=frame)
            captured_frames.append(f"static/frame/{filename}")

    capture.release()

    return {
        "clip_id": clip_id,
        "duration": duration,
        "frames": captured_frames
    }
