from typing import Any, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from PIL import Image

from app.core.exceptions import FrameAnalysisError


class AnalysisService:
    VISION_MODEL_NAME: str = "vikhyatk/moondream2"
    VISION_MODEL_REVISION: str = "2024-08-26"
    REASONING_MODEL_NAME: str = "microsoft/Phi-3-mini-4k-instruct"

    def __init__(self):
        self._vision_model: Optional[Any] = None
        self._vision_tokenizer: Optional[Any] = None
        self._reasoning_model: Optional[Any] = None
        self._reasoning_tokenizer: Optional[Any] = None
        self._device_map = self._get_device_map()

        print(
            f"[INFO] AnalysisService initialized with device: {self._device_map}.")

    def analyze_frames(self, image_paths: list[str]) -> list[str]:
        try:
            self._load_vision_model()

            analysis: list[str] = []

            for image_path in image_paths:
                image = Image.open(image_path)
                print(
                    f"[INFO] Analyzing the image located in the path: {image_path}.")
                # Encoding the image in order to transform it from pixels to numeric vectors so the model can properly process this information.
                encoded_image = self._vision_model.encode_image(image)
                analysis.append(self._vision_model.answer_question(encoded_image,
                                                                   "Describe what is happening in this image.", self._vision_tokenizer))
                print(
                    f"[INFO] Finalized analyzing the image located in the path: {image_path}.")

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

        pipe = pipeline("text-generation", model=self._reasoning_model,
                        tokenizer=self._reasoning_tokenizer)

        args = {
            "max_new_tokens": 100,
            "return_full_text": False,
            "do_sample": False,
            "use_cache": True
        }

        output = pipe(prompt, **args)
        return output[0]["generated_text"]

    def _get_device_map(self) -> dict:
        if torch.cuda.is_available():
            print(
                "[INFO] CUDA is available and therefore it will be used in the device map.")
            return {"": "cuda"}
        elif torch.backends.mps.is_available():
            print(
                "[INFO] MPS is available and therefore it will be used in the device map.")
            return {"": "mps"}
        else:
            print(
                "[INFO] The process will rely in the CPU since CUDA nor MPS is available.")
            return {"": "cpu"}

    def _load_vision_model(self):
        if self._vision_model is None:
            print(
                f"[INFO] Started loading the Vision Model: {self.VISION_MODEL_NAME}.")
            # AutoModelForCausalLM (Automated Model for Causal Language Modeling)
            # Automatically detects and loads the correct Neural Network architecture based on the model name.
            self._vision_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=self.VISION_MODEL_NAME,
                                                                      revision=self.VISION_MODEL_REVISION,
                                                                      trust_remote_code=True,
                                                                      device_map=self._device_map,
                                                                      low_cpu_mem_usage=True,
                                                                      # Using FP16 on Apple Silicon reduces parameter and KV-cache memory footprint,
                                                                      # improving memory bandwidth utilization and speeding up MPS-backed inference.
                                                                      torch_dtype=torch.float16)

            # The Tokenizer acts as the translator converting "human-readable data" to tensors (mathematical arrays) (and vice-versa).
            self._vision_tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=self.VISION_MODEL_NAME, revision=self.VISION_MODEL_REVISION)
            print(
                f"[INFO] Finalized loading the Vision Model: {self.VISION_MODEL_NAME}.")

    def _load_reasoning_model(self):
        if self._reasoning_model is None:
            print(
                f"[INFO] Started loading the Reasoning Model: {self.REASONING_MODEL_NAME}.")
            # AutoModelForCausalLM (Automated Model for Causal Language Modeling)
            # Automatically detects and loads the correct Neural Network architecture based on the model name.
            self._reasoning_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path="microsoft/Phi-3-mini-4k-instruct",
                                                                         device_map=self._device_map,
                                                                         trust_remote_code=True,
                                                                         low_cpu_mem_usage=True,
                                                                         # Using FP16 on Apple Silicon reduces parameter and KV-cache memory footprint,
                                                                         # improving memory bandwidth utilization and speeding up MPS-backed inference.
                                                                         torch_dtype=torch.float16
                                                                         )

            # The Tokenizer acts as the translator converting "human-readable data" to tensors (mathematical arrays) (and vice-versa).
            self._reasoning_tokenizer = AutoTokenizer.from_pretrained(
                "microsoft/Phi-3-mini-4k-instruct")
            print(
                f"[INFO] Finalized loading the Reasoning Model: {self.REASONING_MODEL_NAME}.")
