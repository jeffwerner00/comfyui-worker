# Riley ComfyUI Serverless Worker
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    git wget curl libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN git clone https://github.com/comfyanonymous/ComfyUI.git ComfyUI

# Install ComfyUI requirements (includes aiohttp>=3.11.8, yarl>=1.18.0)
RUN cd ComfyUI && pip install -r requirements.txt

# Force upgrade aiohttp + yarl to latest compatible versions AFTER all installs
# ComfyUI requires aiohttp>=3.11.8 + yarl>=1.18.0; older aiohttp + new yarl = host:port bug
RUN pip install --upgrade 'aiohttp>=3.11.8' 'yarl>=1.18.0'

RUN pip install xformers

RUN cd ComfyUI/custom_nodes && \
    git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git

RUN cd ComfyUI/custom_nodes && \
    git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git && \
    cd comfyui_controlnet_aux && pip install -r requirements.txt

RUN pip install insightface onnxruntime-gpu

RUN pip install runpod==1.6.2

# Final ensure: both aiohttp and yarl at compatible fixed versions
RUN pip install --upgrade 'aiohttp>=3.11.8' 'yarl>=1.18.0'

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

COPY handler.py /workspace/handler.py

ENV PYTHONUNBUFFERED=1
ENV HF_TOKEN=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

CMD ["python3", "-u", "/workspace/handler.py"]