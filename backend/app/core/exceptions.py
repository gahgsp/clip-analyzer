class ClipServiceError(Exception):
    """Base exception for all Clip Service exceptions."""
    pass


class StreamResolutionError(ClipServiceError):
    """Failed to resolve the stream URL."""
    pass


class VideoStreamError(ClipServiceError):
    """Failed to open the video stream."""
    pass


class ClipTooLongError(ClipServiceError):
    """Clip exceeds the maximum duration."""

    def __init__(self, duration: float, max_duration: float):
        self.duration = duration
        self.max_duration = max_duration
        super().__init__(
            f"The clip to be analyzed is too long. Current length is {duration} but maximum allowed is {max_duration}.")


class AnalysisServiceError(Exception):
    """Base exception for all Analysis Service exceptions."""
    pass


class FrameAnalysisError(AnalysisServiceError):
    """Failed to analyze a frame from the video clip."""
    pass
