# comfyui-worker

RunPod serverless worker for ComfyUI SDXL image generation.

## Stack
- ComfyUI (latest)
- Lustify SDXL checkpoint
- LoRA support
- IPAdapter FaceID + InsightFace
- ControlNet OpenPose
- Python 3.11 + CUDA 12.4

## Usage
Accepts ComfyUI workflow JSON via RunPod serverless API. Returns base64-encoded PNG.

### Input
```json
{
  "workflow": { "...ComfyUI workflow nodes..." },
  "client_id": "optional-job-id"
}
```

### Output
```json
{
  "image_b64": "base64-encoded PNG",
  "filename": "output_filename.png",
  "prompt_id": "comfyui-prompt-id"
}
```

## Environment Variables
- `HF_TOKEN` — HuggingFace token for model downloads
- `CIVITAI_TOKEN` — CivitAI token for checkpoint download

## Network Volume
Mount at `/runpod-volume`. Models are downloaded on first run and cached for subsequent workers.
