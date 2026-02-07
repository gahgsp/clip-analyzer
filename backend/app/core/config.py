from pydantic_settings import BaseSettings
from pathlib import Path
from typing import List


class Settings(BaseSettings):
    """Definition of the configuration properties used across the application."""

    # Configuration for the Clip Analysis.
    frames_dir: Path = Path("static/frames")
    frame_percentages: List[float] = [0.10, 0.20,
                                      0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    max_clip_duration: float = 30.0  # The maximum duration in seconds.

    # Configuration for the YTDLP library.
    ytdlp_quiet: bool = True
    ytdlp_noplaylist: bool = True
    ytdlp_nocheckcertificate: bool = True

    # Configuration for the Analysis Service.
    vision_model_name: str = "vikhyatk/moondream2"
    vision_model_revision: str = "2024-08-26"
    reasoning_model_name: str = "microsoft/Phi-3-mini-4k-instruct"


class ClipServiceConfiguration:
    """Configuration for the Clip Service."""

    def __init__(self, settings: Settings):
        self.frames_dir = settings.frames_dir
        self.percentages = settings.frame_percentages
        self.max_duration = settings.max_clip_duration
        self.ytdlp_options = {
            "quiet": settings.ytdlp_quiet,
            "noplaylist": settings.ytdlp_noplaylist,
            # I had to add the "nocheckcertificate" for it to work. For a local environment, it seems to be acceptable to prevent SSL errors.
            "nocheckcertificate": settings.ytdlp_nocheckcertificate,
        }


class AnalysisServiceConfiguration:
    """Configuration for the Analysis Service."""

    def __init__(self, settings: Settings):
        self.vision_model_name = settings.vision_model_name
        self.vision_model_revision = settings.vision_model_revision
        self.reasoning_model_name = settings.reasoning_model_name
