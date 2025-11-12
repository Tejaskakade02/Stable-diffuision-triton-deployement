# Use the official Triton Inference Server image with Python backend support
FROM nvcr.io/nvidia/tritonserver:24.09-py3

# Create working directories
WORKDIR /workspace
RUN mkdir -p /models

# Copy your Triton model repository
COPY model_repository /models

# Install only necessary dependencies for Stable Diffusion
RUN apt-get update && apt-get install -y git && \
    pip install --no-cache-dir \
        torch \
        diffusers \
        transformers \
        pillow \
        numpy \
        accelerate \
        safetensors

# # (Optional) Set Hugging Face cache path inside container
# ENV HF_HOME=/workspace/hf_cache

# Expose Triton ports
EXPOSE 8000 8001 8002

# Run Triton Server with your model repository
CMD ["tritonserver", "--model-repository=/models", "--log-verbose=1"]
