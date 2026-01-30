from pathlib import Path
from typing import List, Dict, Any, Tuple
import cv2
from fastapi import HTTPException
import yt_dlp


class ClipService:
    FRAMES_DIR: Path = Path("static/frames")
    PERCENTAGES: List[float] = [
        0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    # I had to add the "nocheckcertificate" for it to work. For a local environment, it seems to be acceptable to prevent SSL errors.
    OPTIONS_PARAMS: Dict[str, Any] = {
        "quiet": True,
        "noplaylist": True,
        "nocheckcertificate": True,
    }

    def __init__(self):
        # Ensure the directory exists when service is instantiated.
        self.FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    def process_clip(self, url: str) -> Dict[str, Any]:
        clip_id, duration, stream_url = self._resolve_stream_info(url)

        extracted_frames = self._extract_frames(clip_id, duration, stream_url)

        if duration > 30:
            raise HTTPException(
                status_code=400, detail=f"The clip to be analyzed is too long. Maximum length allowed is 30 seconds.")

        return {
            "clip_id": clip_id,
            "duration": duration,
            "frames": extracted_frames
        }

    def _resolve_stream_info(self, url: str) -> Tuple[str, float, str]:
        with yt_dlp.YoutubeDL(params=self.OPTIONS_PARAMS) as ydl:
            info = ydl.extract_info(url=url, download=False)

            stream_url: str = info.get("url", "")
            duration: float = float(info.get("duration", 0.0))
            clip_id: str = info.get("id", "unknown")

            if not stream_url:
                raise HTTPException(status_code=500,
                                    detail=f"Failed to resolve the Twitch URL.")

            return clip_id, duration, stream_url

    def _extract_frames(self, clip_id: str, duration: float, stream_url: str) -> List[str]:
        captured_frames: List[str] = []
        capture = cv2.VideoCapture(filename=stream_url)

        if not capture.isOpened():
            raise HTTPException(
                status_code=500, detail=f"Could not open the video stream from: {str(stream_url)}.")

        try:
            for percentage in self.PERCENTAGES:
                timestamp_in_ms = (duration * percentage) * 1000
                capture.set(propId=cv2.CAP_PROP_POS_MSEC,
                            value=timestamp_in_ms)

                success, frame = capture.read()
                if success:
                    filename = f"{clip_id}_{int(percentage * 100)}.jpg"
                    filepath = self.FRAMES_DIR / filename

                    cv2.imwrite(filename=str(filepath), img=frame)

                    captured_frames.append(f"static/frames/{filename}")
        finally:
            # Ensure that the resources are released whether it was successful or had an error.
            capture.release()

        return captured_frames
