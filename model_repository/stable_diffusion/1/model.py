import torch
import numpy as np
import io
import base64
from diffusers import StableDiffusionPipeline
from PIL import Image
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    """
    Triton Python Backend model for Stable Diffusion.
    - Accepts a text prompt
    - Generates an image using a lightweight Stable Diffusion model
    - Returns a base64-encoded PNG string
    """

    def initialize(self, args):
        """Called once when the model is loaded."""
        # Use a small and fast model
        self.model_name = "stabilityai/sd-turbo"   # or "segmind/tiny-sd"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load the Stable Diffusion model
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)

        print(f"[INFO] {self.model_name} loaded on {self.device}")

    def execute(self, requests):
        """Called for every inference request."""
        responses = []
        for request in requests:
            prompt_input = pb_utils.get_input_tensor_by_name(request, "prompt")
            prompt = prompt_input.as_numpy()[0].item().decode("utf-8")

            print(f"[INFO] Generating image for: {prompt}")

            with torch.inference_mode():
                image = self.pipe(prompt, num_inference_steps=1).images[0]

            # Convert to base64
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            out_tensor = pb_utils.Tensor("generated_image", np.array([img_b64], dtype=object))
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))

        return responses

    def finalize(self):
        print("[INFO] Model shutdown complete.")
