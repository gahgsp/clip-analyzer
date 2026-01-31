from functools import lru_cache
from app.service.clip_service import ClipService
from app.service.analysis_service import AnalysisService


def get_clip_service() -> ClipService:
    return ClipService()


@lru_cache
def get_analysis_service() -> AnalysisService:
    """
    As AnalysisService loads a large ML model during initialization, this dependency will be cached to act as a "singleton".

    FastAPI will call this function only once per worker process and it will reuse the same instance from AnalysisService
    for all incoming requests from the API.

    The cache created by it is basically a dict that lives in the RAM inside the Python process and it is garbage-collected as well in the process finished.
    """
    return AnalysisService()
