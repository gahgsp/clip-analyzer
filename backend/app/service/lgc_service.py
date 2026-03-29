import logging

from app.core.config import AnalysisServiceConfiguration, ClipServiceConfiguration
from app.service.clip_service import ClipService
from app.service.analysis_service import AnalysisService
from langchain_core.runnables import RunnableLambda

from app.model.clip import ClipResponse, FrameAnalysis, ProcessedClip

logger = logging.getLogger(__name__)


class LangChainService:
    """
    After reading the Chapter 7. Advanced Text Generation Techniques and Tools from the book
    Hands-On Large Language Models: Language Understanding and Generation which briefly introduced about LangChain,
    I decided to apply it in this project.

    It is also possible to see how the existing project's architecture played well since there was no need to change any existing code for the core of the processes.
    We are basically just calling them with a few additions for domain constants required to use LangChain.
    """

    def __init__(self, analysis_configuration: AnalysisServiceConfiguration, clip_configuration: ClipServiceConfiguration):
        self.clip_service = ClipService(configuration=clip_configuration)
        self.analysis_service = AnalysisService(
            configuration=analysis_configuration)

        # A RunnableLambda is a piece of logic written in Python that executes a specific function.
        # It is not a LLM call and it converts the function in the parameter to a Runnable to make it composable to build chains.
        _process_clip = RunnableLambda(
            func=self._process_clip_step, name="process_clip")

        _analyze_frames = RunnableLambda(
            self._analyze_frames_step, name="analyze_frames"
        )

        _generate_summary = RunnableLambda(
            self._generate_summary_step, name="generate_summary"
        )

        # We use the pipe operator (|) to build the chain.
        # The order of execution is from the left to the right.
        self.chain = _process_clip | _analyze_frames | _generate_summary

        logger.info("The LangChain 'LangChainService' is initialized.")

    def _process_clip_step(self, url: str) -> ProcessedClip:
        """Step 1: We download the clip from the URL received as parameter."""
        logger.info(f"Step 1 from Chain: processing clip from URL: {url}.")
        return self.clip_service.process_clip(url=url)

    def _analyze_frames_step(self, clip: ProcessedClip) -> dict:
        """
        Step 2: We run the vision model through all the extracted frames.

        We needed to return a "custom" dictionary to avoid losing the metadata of the clip in
        the middle of the chain. 
        This is important since we use it in the end for the final response.
        """
        logger.info(
            f"Step 2 from Chain: analyzing {len(clip.frame_paths)} frames.")
        descriptions = self.analysis_service.analyze_frames(clip.frame_paths)

        return {
            "clip": clip,
            "descriptions": descriptions,
        }

    def _generate_summary_step(self, data: dict) -> ClipResponse:
        """
        Step 3: Summarize all frame descriptions into a single summary.
        """
        clip: ProcessedClip = data["clip"]
        descriptions: list[str] = data["descriptions"]

        logger.info("Step 3 from Chain: generating summary.")
        summary = self.analysis_service.generate_summary(descriptions)

        frames = [
            FrameAnalysis(path=path, description=desc)
            for path, desc in zip(clip.frame_paths, descriptions)
        ]

        return ClipResponse(
            clip_id=clip.clip_id,
            duration=clip.duration,
            frames=frames,
            summary=summary,
        )
