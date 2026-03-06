# Riley ComfyUI Serverless Worker
# Build trigger: 2026-03-06
# Base: runpod/pytorch - pre-configured Python + CUDA + RunPod SDK environment

FROM runpod/pytorch:2.6.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y \
    git wget curl libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Clone ComfyUI
RUN git clone https://github.com/comfyanonymous/ComfyUI.git ComfyUI

# Install ComfyUI requirements (uses base image's Python env)
RUN cd ComfyUI && pip install -r requirements.txt

# Install xformers compatible with torch 2.6.0 (let pip resolve)
RUN pip install xformers

# Install custom nodes
RUN cd ComfyUI/custom_nodes && \
    git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git

RUN cd ComfyUI/custom_nodes && \
    git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git && \
    cd comfyui_controlnet_aux && pip install -r requirements.txt

# Install InsightFace + ONNX
RUN pip install insightface onnxruntime-gpu

# Pin RunPod SDK to known-working version
RUN pip install runpod==1.6.2

# Create model directories (will be symlinked to network volume at runtime)
RUN mkdir -p /workspace/ComfyUI/models/checkpoints \
    /workspace/ComfyUI/models/loras \
    /workspace/ComfyUI/models/ipadapter \
    /workspace/ComfyUI/models/clip_vision \
    /workspace/ComfyUI/models/controlnet \
    /workspace/ComfyUI/models/insightface/models/buffalo_l \
    /workspace/ComfyUI/input \
    /workspace/ComfyUI/output \
    /runpod-volume/models \
    /runpod-volume/inputs \
    /runpod-volume/insightface

# Copy handler
COPY handler.py /workspace/handler.py

ENV PYTHONUNBUFFERED=1
ENV HF_TOKEN=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

CMD ["python3", "-u", "/workspace/handler.py"]