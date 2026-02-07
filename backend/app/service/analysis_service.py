import logging
from typing import Any, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

from app.core.exceptions import FrameAnalysisError
from app.core.config import AnalysisServiceConfiguration

logger = logging.getLogger(__name__)


class AnalysisService:
    def __init__(self, configuration: AnalysisServiceConfiguration):
        self.vision_model_name = configuration.vision_model_name
        self.vision_model_revision = configuration.vision_model_revision
        self.reasoning_model_name = configuration.reasoning_model_name

        self._vision_model: Optional[Any] = None
        self._vision_tokenizer: Optional[Any] = None
        self._reasoning_model: Optional[Any] = None
        self._reasoning_tokenizer: Optional[Any] = None
        self._device_map = self._get_device_map()

        logger.info(
            f"AnalysisService initialized with device: {self._device_map}.")

    def analyze_frames(self, image_paths: list[str]) -> list[str]:
        try:
            self._load_vision_model()

            analysis: list[str] = []

            for image_path in image_paths:
                image = Image.open(image_path)
                logger.info(
                    f"Analyzing the image located in the path: {image_path}.")
                # Encoding the image in order to transform it from pixels to numeric vectors so the model can properly process this information.
                encoded_image = self._vision_model.encode_image(image)
                analysis.append(self._vision_model.answer_question(encoded_image,
                                                                   "Describe what is happening in this image.", self._vision_tokenizer))
                logger.info(
                    f"Finalized analyzing the image located in the path: {image_path}.")

            return analysis
        except FileNotFoundError as e:
            raise FrameAnalysisError(
                f"Could not find an image in the following path: {image_paths}.")
        except Exception as e:
            raise FrameAnalysisError(
                f"An error occurred while analyzing the frame {image_paths}: {str(e)}.")

    def generate_summary(self, descriptions: list[str]) -> str:
        self._load_reasoning_model()

        context = " ".join(
            [f"Event: {i + 1}: {description}" for i, description in enumerate(descriptions)]).strip()

        prompt = f"<|user|>The following is a compiled list of descriptions from frames extracted from a video clip. They are ordered in the same order that the events happened in the video clip. Summary all the descriptions into a single cohesive 2-sentence story about what happened in the clip: {context}<|end|><|assistant|>"

        inputs = self._reasoning_tokenizer(
            prompt, return_tensors="pt").to(self._reasoning_model.device)

        logger.info(
            "Starting the process of generating a summary for the descriptions.")
        with torch.inference_mode():
            output_ids = self._reasoning_model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=False,
                use_cache=True,
                eos_token_id=self._reasoning_tokenizer.eos_token_id,
            )
        logger.info(
            "Finished the process of generating a summary for descriptions.")

        generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]

        return self._reasoning_tokenizer.decode(
            generated_ids, skip_special_tokens=True
        ).strip()

    def _get_device_map(self) -> dict:
        if torch.cuda.is_available():
            logger.info(
                "CUDA is available and therefore it will be used in the device map.")
            return {"": "cuda"}
        elif torch.backends.mps.is_available():
            logger.info(
                "MPS is available and therefore it will be used in the device map.")
            return {"": "mps"}
        else:
            logger.info(
                "The process will rely in the CPU since CUDA nor MPS is available.")
            return {"": "cpu"}

    def _load_vision_model(self):
        if self._vision_model is None:
            logger.info(
                "Started loading the Vision Model: {self.vision_model_name}.")
            # AutoModelForCausalLM (Automated Model for Causal Language Modeling)
            # Automatically detects and loads the correct Neural Network architecture based on the model name.
            self._vision_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=self.vision_model_name,
                                                                      revision=self.vision_model_revision,
                                                                      trust_remote_code=True,
                                                                      device_map=self._device_map,
                                                                      low_cpu_mem_usage=True,
                                                                      # Using FP16 on Apple Silicon reduces parameter and KV-cache memory footprint,
                                                                      # improving memory bandwidth utilization and speeding up MPS-backed inference.
                                                                      torch_dtype=torch.float16)

            # The Tokenizer acts as the translator converting "human-readable data" to tensors (mathematical arrays) (and vice-versa).
            self._vision_tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=self.vision_model_name, revision=self.vision_model_revision)
            logger.info(
                f"Finalized loading the Vision Model: {self.vision_model_name}.")

    def _load_reasoning_model(self):
        if self._reasoning_model is None:
            logger.info(
                f"Started loading the Reasoning Model: {self.reasoning_model_name}.")
            # AutoModelForCausalLM (Automated Model for Causal Language Modeling)
            # Automatically detects and loads the correct Neural Network architecture based on the model name.
            self._reasoning_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=self.reasoning_model_name,
                                                                         device_map=self._device_map,
                                                                         trust_remote_code=True,
                                                                         low_cpu_mem_usage=True,
                                                                         # Using FP16 on Apple Silicon reduces parameter and KV-cache memory footprint,
                                                                         # improving memory bandwidth utilization and speeding up MPS-backed inference.
                                                                         torch_dtype=torch.float16
                                                                         )

            # The Tokenizer acts as the translator converting "human-readable data" to tensors (mathematical arrays) (and vice-versa).
            self._reasoning_tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=self.reasoning_model_name)
            logger.info(
                f"Finalized loading the Reasoning Model: {self.reasoning_model_name}.")
