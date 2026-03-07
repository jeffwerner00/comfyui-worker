# Riley ComfyUI Serverless Worker
# Base: runpod/pytorch - pre-configured Python + CUDA + RunPod SDK environment

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y \
    git wget curl libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Clone ComfyUI
RUN git clone https://github.com/comfyanonymous/ComfyUI.git ComfyUI

# Pin yarl/aiohttp BEFORE installing ComfyUI requirements
# yarl>=1.17 rejects '127.0.0.1:8188' as a host (colon not allowed) - breaks ComfyUI request handling
RUN pip install 'yarl<1.17.0' 'aiohttp<3.11.0'

# Install ComfyUI requirements
RUN cd ComfyUI && pip install -r requirements.txt

# Re-pin yarl/aiohttp after requirements (in case they get upgraded)
RUN pip install 'yarl<1.17.0' 'aiohttp<3.11.0'

# Install xformers
RUN pip install xformers

# Install custom nodes
RUN cd ComfyUI/custom_nodes && \
    git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git

RUN cd ComfyUI/custom_nodes && \
    git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git && \
    cd comfyui_controlnet_aux && pip install -r requirements.txt

# Install InsightFace + ONNX
RUN pip install insightface onnxruntime-gpu

# Pin RunPod SDK
RUN pip install runpod==1.6.2

# FINAL yarl/aiohttp pin - must be last pip install
# xformers and insightface can re-upgrade yarl; this ensures we end up pinned
RUN pip install 'yarl<1.17.0' 'aiohttp<3.11.0'

# Create model directories
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