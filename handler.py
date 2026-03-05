"""
RunPod Serverless Handler for Riley ComfyUI Pipeline
Accepts ComfyUI workflow JSON, runs generation, returns base64 image
"""

import runpod
import json
import os
import time
import urllib.request
import urllib.error
import base64
import subprocess
import threading
import sys

COMFY_DIR = "/workspace/ComfyUI"
MODEL_VOLUME = "/runpod-volume"
COMFY_HOST = "http://127.0.0.1:8188"

# Model definitions — checked/downloaded on startup
MODELS = {
    "checkpoints/lustifySDXLv7.safetensors": {
        "url": "https://civitai.com/api/download/models/2155386",
        "auth": f"Bearer {os.environ.get('CIVITAI_TOKEN', '')}",
        "size_gb": 6.9
    },
    "loras/riley-pony-v1.safetensors": {
        "url": "https://huggingface.co/datasets/Jwerner00/unmasked-assets/resolve/main/riley-pony-v1.safetensors",
        "auth": f"Bearer {os.environ.get('HF_TOKEN', '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')}",
        "size_gb": 0.14
    },
    "ipadapter/ip-adapter-faceid_sdxl.bin": {
        "url": "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sdxl.bin",
        "auth": f"Bearer {os.environ.get('HF_TOKEN', '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')}",
        "size_gb": 1.0
    },
    "clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors": {
        "url": "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors",
        "auth": f"Bearer {os.environ.get('HF_TOKEN', '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')}",
        "size_gb": 2.5
    },
    "controlnet/control-lora-openposeXL2-rank256.safetensors": {
        "url": "https://huggingface.co/thibaud/controlnet-openpose-sdxl-1.0/resolve/main/control-lora-openposeXL2-rank256.safetensors",
        "auth": f"Bearer {os.environ.get('HF_TOKEN', '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')}",
        "size_gb": 0.74
    }
}

# InsightFace buffalo_l models
INSIGHTFACE_MODELS = [
    "1k3d68.onnx",
    "2d106det.onnx",
    "det_10g.onnx",
    "genderage.onnx",
    "w600k_r50.onnx"
]

def log(msg):
    print(f"[handler] {msg}", flush=True)

def ensure_models():
    """Download missing models to network volume, symlink into ComfyUI"""
    log("Checking models...")
    
    volume_models = os.path.join(MODEL_VOLUME, "models")
    comfy_models = os.path.join(COMFY_DIR, "models")
    
    for rel_path, info in MODELS.items():
        vol_path = os.path.join(volume_models, rel_path)
        comfy_path = os.path.join(comfy_models, rel_path)
        
        os.makedirs(os.path.dirname(vol_path), exist_ok=True)
        os.makedirs(os.path.dirname(comfy_path), exist_ok=True)
        
        # Download if not on volume
        if not os.path.exists(vol_path) or os.path.getsize(vol_path) < 1_000_000:
            log(f"Downloading {rel_path} (~{info['size_gb']}GB)...")
            cmd = ["wget", "-q", "--show-progress", "-O", vol_path]
            if info.get("auth"):
                cmd += ["--header", f"Authorization: {info['auth']}"]
            cmd.append(info["url"])
            result = subprocess.run(cmd, timeout=1800)
            if result.returncode != 0:
                log(f"wget failed for {rel_path}, trying curl...")
                curl_cmd = ["curl", "-L", "-o", vol_path]
                if info.get("auth"):
                    curl_cmd += ["-H", f"Authorization: {info['auth']}"]
                curl_cmd.append(info["url"])
                subprocess.run(curl_cmd, timeout=1800, check=True)
            log(f"Downloaded: {rel_path}")
        
        # Symlink into ComfyUI models dir
        if not os.path.exists(comfy_path):
            os.symlink(vol_path, comfy_path)
            log(f"Symlinked: {rel_path}")
    
    # InsightFace buffalo_l
    insightface_vol = os.path.join(MODEL_VOLUME, "insightface/models/buffalo_l")
    insightface_comfy = os.path.join(COMFY_DIR, "models/insightface/models/buffalo_l")
    os.makedirs(insightface_vol, exist_ok=True)
    os.makedirs(insightface_comfy, exist_ok=True)
    
    for model_file in INSIGHTFACE_MODELS:
        vol_file = os.path.join(insightface_vol, model_file)
        comfy_file = os.path.join(insightface_comfy, model_file)
        if not os.path.exists(vol_file) or os.path.getsize(vol_file) < 10_000:
            log(f"InsightFace {model_file} missing — will be auto-downloaded by insightface package on first use")
        if not os.path.exists(comfy_file) and os.path.exists(vol_file):
            os.symlink(vol_file, comfy_file)
    
    # Face reference image
    face_ref_vol = os.path.join(MODEL_VOLUME, "inputs/riley-face-ref.jpg")
    face_ref_comfy = os.path.join(COMFY_DIR, "input/riley-face-ref.jpg")
    if not os.path.exists(face_ref_comfy) and os.path.exists(face_ref_vol):
        os.symlink(face_ref_vol, face_ref_comfy)
    
    log("Model check complete")

def start_comfyui():
    """Start ComfyUI in background"""
    log("Starting ComfyUI...")
    proc = subprocess.Popen(
        [sys.executable, "main.py", "--listen", "0.0.0.0", "--port", "8188"],
        cwd=COMFY_DIR,
        stdout=open("/workspace/comfyui.log", "w"),
        stderr=subprocess.STDOUT
    )
    
    # Wait for ComfyUI to be ready
    for i in range(60):
        try:
            urllib.request.urlopen(f"{COMFY_HOST}/system_stats", timeout=5)
            log(f"ComfyUI ready after {i+1}s")
            return proc
        except:
            time.sleep(1)
    
    raise RuntimeError("ComfyUI failed to start")

def queue_workflow(workflow: dict, client_id: str = "serverless") -> str:
    """Submit workflow to ComfyUI, return prompt_id"""
    data = json.dumps({"prompt": workflow, "client_id": client_id}).encode()
    req = urllib.request.Request(
        f"{COMFY_HOST}/prompt",
        data=data,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())
    
    if result.get("node_errors"):
        raise ValueError(f"Node errors: {json.dumps(result['node_errors'])}")
    
    return result["prompt_id"]

def wait_for_result(prompt_id: str, timeout: int = 300) -> dict:
    """Poll history until generation complete, return output info"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(f"{COMFY_HOST}/history/{prompt_id}", timeout=10) as resp:
                history = json.loads(resp.read())
            
            if prompt_id in history:
                status = history[prompt_id].get("status", {})
                if status.get("status_str") == "error":
                    raise RuntimeError(f"Generation error: {status.get('messages')}")
                
                for node_id, output in history[prompt_id].get("outputs", {}).items():
                    for img in output.get("images", []):
                        return img
        except urllib.error.URLError:
            pass
        time.sleep(2)
    
    raise TimeoutError(f"Generation timed out after {timeout}s")

def download_image(filename: str) -> bytes:
    """Download generated image from ComfyUI"""
    url = f"{COMFY_HOST}/view?filename={urllib.parse.quote(filename)}&subfolder=&type=output"
    with urllib.request.urlopen(url, timeout=30) as resp:
        return resp.read()

import urllib.parse

# --- Startup ---
_comfyui_proc = None

def initialize():
    global _comfyui_proc
    ensure_models()
    _comfyui_proc = start_comfyui()

# --- Handler ---
def handler(job):
    """
    Job input:
    {
        "workflow": { ...ComfyUI workflow JSON... },
        "client_id": "optional-string"
    }
    
    Returns:
    {
        "image_b64": "base64 encoded PNG",
        "filename": "generated_filename.png"
    }
    """
    try:
        job_input = job["input"]
        workflow = job_input.get("workflow")
        client_id = job_input.get("client_id", f"job-{job.get('id', 'unknown')}")
        
        if not workflow:
            return {"error": "No workflow provided"}
        
        log(f"Processing job {client_id}")
        
        prompt_id = queue_workflow(workflow, client_id)
        log(f"Queued: {prompt_id}")
        
        img_info = wait_for_result(prompt_id)
        log(f"Generated: {img_info['filename']}")
        
        img_bytes = download_image(img_info["filename"])
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        
        return {
            "image_b64": img_b64,
            "filename": img_info["filename"],
            "prompt_id": prompt_id
        }
    
    except Exception as e:
        log(f"Error: {e}")
        return {"error": str(e)}

# Initialize on module load (RunPod calls this before first job)
initialize()

runpod.serverless.start({"handler": handler})
