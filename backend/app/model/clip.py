from pydantic import BaseModel
from typing import List


class ClipRequest(BaseModel):
    url: str


class FrameAnalysis(BaseModel):
    path: str
    description: str


class ClipResponse(BaseModel):
    clip_id: str
    duration: float
    frames: List[FrameAnalysis]
