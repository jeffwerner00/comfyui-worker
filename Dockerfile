# Riley ComfyUI Serverless Worker
# Base: CUDA 12.4 + Python 3.11

FROM runpod/base:1.0.3-cuda1281-ubuntu2204

# System deps
RUN apt-get update && apt-get install -y \
    git wget curl libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    python3-pip python3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python

WORKDIR /workspace

# Clone ComfyUI
RUN git clone https://github.com/comfyanonymous/ComfyUI.git ComfyUI

# Install ComfyUI requirements
RUN cd ComfyUI && python3 -m pip install -r requirements.txt

# Install torch stack (CUDA 12.4 compatible)
RUN python3 -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Install xformers
RUN python3 -m pip install xformers==0.0.29.post2

# Install custom nodes
RUN cd ComfyUI/custom_nodes && \
    git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git

RUN cd ComfyUI/custom_nodes && \
    git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git && \
    cd comfyui_controlnet_aux && python3 -m pip install -r requirements.txt

# Install InsightFace + ONNX
RUN python3 -m pip install insightface onnxruntime-gpu

# Install RunPod SDK
RUN python3 -m pip install runpod

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
