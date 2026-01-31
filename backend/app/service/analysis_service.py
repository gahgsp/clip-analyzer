import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

from app.core.exceptions import FrameAnalysisError


class AnalysisService:
    MODEL_NAME: str = "vikhyatk/moondream2"
    MODEL_REVISION: str = "2024-08-26"

    def __init__(self):
        # AutoModelForCausalLM (Automated Model for Causal Language Modeling)
        # Automatically detects and loads the correct Neural Network architecture based on the model name.
        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=self.MODEL_NAME,
                                                          revision=self.MODEL_REVISION,
                                                          trust_remote_code=True,
                                                          device_map=self._get_device_map())
        print(
            f"[INFO] Initializing with [model]: {self.MODEL_NAME} and [revision]: {self.MODEL_REVISION}.")

        # The Tokenizer acts as the translator converting "human-readable data" to tensors (mathematical arrays) (and vice-versa).
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.MODEL_NAME, revision=self.MODEL_REVISION)

    def analyze_frame(self, image_path: str) -> str:
        try:
            image = Image.open(image_path)
            print(f"[INFO] Analyzing image located in the path: {image_path}.")
            # Encoding the image in order to transform it from pixels to numeric vectors so the model can properly process this information.
            encoded_image = self.model.encode_image(image)

            return self.model.answer_question(encoded_image,
                                              "Describe what is happening in this image.", self.tokenizer)
        except FileNotFoundError as e:
            raise FrameAnalysisError(
                f"Could not find an image in the following path: {image_path}.")
        except Exception as e:
            raise FrameAnalysisError(
                f"An error occurred while analyzing the frame {image_path}: {str(e)}.")

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
