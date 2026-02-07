from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from app.service.clip_service import ClipService
from app.service.analysis_service import AnalysisService
from app.core.config import AnalysisServiceConfiguration, ClipServiceConfiguration, Settings


@lru_cache
def get_settings() -> Settings:
    return Settings()


def get_clip_service_configuration(settings: Annotated[Settings, Depends(get_settings)]) -> ClipServiceConfiguration:
    if settings is None:
        settings = get_settings()
    return ClipServiceConfiguration(settings=settings)


def get_clip_service(configuration: Annotated[ClipServiceConfiguration, Depends(get_clip_service_configuration)]) -> ClipService:
    if configuration is None:
        configuration = get_clip_service_configuration()
    return ClipService(configuration=configuration)


def get_analysis_service_configuration(settings: Annotated[Settings, Depends(get_settings)]) -> AnalysisServiceConfiguration:
    if settings is None:
        settings = get_settings()
    return AnalysisServiceConfiguration(settings=settings)


@lru_cache
def get_analysis_service(configuration: Annotated[AnalysisServiceConfiguration, Depends(get_analysis_service_configuration)]) -> AnalysisService:
    """
    As AnalysisService loads a large ML model during initialization, this dependency will be cached to act as a "singleton".

    FastAPI will call this function only once per worker process and it will reuse the same instance from AnalysisService
    for all incoming requests from the API.

    The cache created by it is basically a dict that lives in the RAM inside the Python process and it is garbage-collected as well in the process finished.
    """
    if configuration is None:
        configuration = get_analysis_service_configuration()
    return AnalysisService(configuration=configuration)
