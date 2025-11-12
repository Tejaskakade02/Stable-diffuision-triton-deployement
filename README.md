# 🧠 Stable Diffusion Turbo — Triton Inference Deployment

Deploy **Stable Diffusion Turbo (stabilityai/sd-turbo)** using **NVIDIA Triton Inference Server** with the **Python backend**. This setup provides a lightweight, GPU-optimized image generation server capable of real-time text-to-image inference.

---

## 📁 Project Structure

```
lora-deployment/
├── Dockerfile
├── model_repository/
│   └── stable_diffusion/
│       ├── 1/
│       │   └── model.py
│       └── config.pbtxt
└── test_S_D.py
```

---

## 🚀 Features

✅ Uses **Stable Diffusion Turbo** — fast, lightweight (~2 GB)

✅ Deployed using **Triton Inference Server (v2.50.0)**

✅ Full **GPU acceleration with CUDA 12.x**

✅ Returns **images as Base64-encoded PNGs**

✅ Simple **HTTP client** for testing

---

## ⚙️ Prerequisites 🧩

### On your host (Windows + WSL2 recommended)

* NVIDIA GPU + latest driver (≥ v535)
* WSL2 with Ubuntu (recommended for full CUDA support)
* Docker and NVIDIA Container Toolkit installed

Verify GPU setup:

```bash
docker run --gpus all nvidia/cuda:12.3.0-base-ubuntu22.04 nvidia-smi
```

You should see your GPU listed.

---

## 🐳 Building the Triton Docker Image

From your project directory:

```bash
docker build -t triton-sd-turbo .
```

---

## 🧠 Dockerfile Overview

This image:

* Uses **NVIDIA Triton Server** with Python backend
* Installs only essential dependencies:

  * torch
  * diffusers
  * transformers
  * pillow
  * numpy
  * safetensors

### Key Dockerfile snippet

```dockerfile
FROM nvcr.io/nvidia/tritonserver:24.09-py3

WORKDIR /workspace
RUN mkdir -p /models
COPY model_repository /models

RUN apt-get update && apt-get install -y git && \
    pip install --no-cache-dir torch diffusers transformers pillow numpy safetensors

EXPOSE 8000 8001 8002
CMD ["tritonserver", "--model-repository=/models", "--log-verbose=1"]
```

---

## ▶️ Running the Triton Server

```bash
docker run --gpus all -it --rm \
-p 8000:8000 -p 8001:8001 -p 8002:8002 \
-v "$(pwd)/model_repository:/models" \
triton-sd-turbo
```

If successful, you’ll see logs like:

```
[INFO] stabilityai/sd-turbo loaded on cuda
Triton Server Ready
```

---

## 🧪 Testing the Model

Run the included test client:

```bash
python3 test_S_D.py
```

Example Output:

```
[INFO] Sending prompt: a futuristic red sports car parked on Mars, ultra-detailed, cinematic lighting
[INFO] Inference successful!
[✅] Image saved as 'output.png'
```

Your generated image will appear as `output.png`.

---

## 🧩 Example Prompt

Try creative inputs:

```python
prompt = "a medieval castle floating in the clouds, cinematic lighting"
```

Or modify the `test_S_D.py` file to generate your own ideas.

---

## 🧰 Troubleshooting

| Issue                                                            | Cause                             | Fix                                                     |
| ---------------------------------------------------------------- | --------------------------------- | ------------------------------------------------------- |
| ImportError: transformers not found                              | Missing dependency                | Add `transformers` to pip install in Dockerfile         |
| AttributeError: 'numpy.ndarray' object has no attribute 'decode' | Incorrect decoding in model.py    | Use `.item().decode("utf-8")`                           |
| Triton exits immediately                                         | Model load failure                | Check logs with `docker logs <container_id>`            |
| No container visible                                             | It exited on error                | Run `docker ps -a` and inspect logs                     |
| GPU not detected                                                 | WSL/Docker not configured for GPU | Reinstall `nvidia-container-toolkit` and restart Docker |

---

## 📄 model.py (core logic)

```python
prompt = prompt_input.as_numpy()[0].item().decode("utf-8")
image = self.pipe(prompt, num_inference_steps=1).images[0]
```

✅ Returns a **base64 PNG string** as the model output.

---

## 🧠 Model Information

| Model                  | Repo                 | Size    | VRAM   | Description                                         |
| ---------------------- | -------------------- | ------- | ------ | --------------------------------------------------- |
| Stable Diffusion Turbo | stabilityai/sd-turbo | ~2 GB   | 3–4 GB | Fast, distilled version of SD2.1 (1-step inference) |
| Tiny SD                | segmind/tiny-sd      | ~1.3 GB | 2–3 GB | Smaller, for CPU/edge testing                       |

---

## 🧹 Cleanup

To stop and remove containers:

```bash
docker ps -a
docker rm $(docker ps -aq)
```

To remove the image:

```bash
docker rmi triton-sd-turbo
```

---

## 🏁 Summary

✅ Lightweight Stable Diffusion Turbo
✅ Deployed with NVIDIA Triton Server
✅ GPU-accelerated inference
✅ Simple HTTP client for testing

This setup gives you a **fully self-contained, production-ready deployment pipeline** for **text-to-image inference**.
