from fastapi import HTTPException
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image


class AnalysisService:
    def __init__(self):
        MODEL_NAME: str = "vikhyatk/moondream2"
        MODEL_REVISION: str = "2024-08-26"

        # AutoModelForCausalLM (Automated Model for Causal Language Modeling)
        # Automatically detects and loads the correct Neural Network architecture based on the model name.
        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=MODEL_NAME,
                                                          revision=MODEL_REVISION,
                                                          trust_remote_code=True,
                                                          device_map={"": "mps"})

        # The Tokenizer acts as the translator converting "human-readable data" to tensors (mathematical arrays) (and vice-versa).
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=MODEL_NAME, revision=MODEL_REVISION)

    def analyze_frame(self, image_path: str) -> str:
        try:
            image = Image.open(image_path)
            # Encoding the image in order to transform it from pixels to numeric vectors so the model can properly process this information.
            encoded_image = self.model.encode_image(image)

            return self.model.answer_question(encoded_image,
                                              "Describe what is happening in this image.", self.tokenizer)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"An error occurred while analyzing the frame {image_path}: {str(e)}.")
