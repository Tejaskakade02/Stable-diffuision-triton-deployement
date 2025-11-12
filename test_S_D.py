import base64
import json
import io
import requests
from PIL import Image

# Triton Inference Server HTTP endpoint
TRITON_URL = "http://localhost:8000/v2/models/stable_diffusion/infer"

# 🖼️ Your text prompt
prompt = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k"

# Prepare request payload for Triton
payload = {
    "inputs": [
        {
            "name": "prompt",
            "shape": [1, 1],
            "datatype": "BYTES",
            "data": [prompt.encode("utf-8").decode("utf-8")]
        }
    ]
}

print(f"[INFO] Sending prompt: {prompt}")

# Send request to Triton
response = requests.post(TRITON_URL, json=payload)

# Handle response
if response.status_code != 200:
    print(f"[ERROR] Triton inference failed: {response.status_code}")
    print(response.text)
else:
    print("[INFO] Inference successful!")

    # Parse JSON output
    result = response.json()
    img_base64 = result["outputs"][0]["data"][0]

    # Decode and save image
    image_data = base64.b64decode(img_base64)
    image = Image.open(io.BytesIO(image_data))
    image.save("output.png")

    print("[✅] Image saved as 'output.png'")
