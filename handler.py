"""
RunPod Serverless Handler for Riley/Pam/Vera/Nova ComfyUI Pipeline
"""
import runpod, json, os, time, urllib.request, urllib.error, urllib.parse, base64, subprocess, threading, sys

COMFY_DIR = "/workspace/ComfyUI"
MODEL_VOLUME = "/runpod-volume"
COMFY_HOST = "http://127.0.0.1:8188"
CIVITAI_TOKEN = os.environ.get('CIVITAI_TOKEN', 'd0933fef3ae3fee1f474425e26266057')
HF_TOKEN_VAL = os.environ.get('HF_TOKEN', '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')

MODELS = {
    "checkpoints/lustifySDXLv7.safetensors": {
        "url": f"https://civitai.com/api/download/models/2155386?token={CIVITAI_TOKEN}",
        "size_gb": 6.9
    },
    "checkpoints/ponyDiffusionV6XL.safetensors": {
        "url": "https://huggingface.co/LyliaEngine/Pony_Diffusion_V6_XL/resolve/main/ponyDiffusionV6XL_v6StartWithThisOne.safetensors",
        "size_gb": 6.46
    },
    "loras/riley-pony-v1.safetensors": {
        "url": "https://huggingface.co/datasets/Jwerner00/unmasked-assets/resolve/main/riley-pony-v1.safetensors",
        "auth": f"Bearer {HF_TOKEN_VAL}", "size_gb": 0.14
    },
    "loras/pam-pony-v2.safetensors": {
        "url": "https://huggingface.co/datasets/Jwerner00/unmasked-assets/resolve/main/pam_pony_v2.safetensors",
        "auth": f"Bearer {HF_TOKEN_VAL}", "size_gb": 0.14
    },
    "loras/vera-pony-v2.safetensors": {
        "url": "https://huggingface.co/datasets/Jwerner00/unmasked-assets/resolve/main/vera_pony_v2.safetensors",
        "auth": f"Bearer {HF_TOKEN_VAL}", "size_gb": 0.14
    },
    "loras/nova-pony-v2.safetensors": {
        "url": "https://huggingface.co/datasets/Jwerner00/unmasked-assets/resolve/main/nova_pony_v2.safetensors",
        "auth": f"Bearer {HF_TOKEN_VAL}", "size_gb": 0.14
    },
    "loras/rowan-pony-v2.safetensors": {
        "url": "https://huggingface.co/datasets/Jwerner00/unmasked-assets/resolve/main/rowan_pony_v2.safetensors",
        "auth": f"Bearer {HF_TOKEN_VAL}", "size_gb": 0.23
    },
    "loras/reed-pony-v2.safetensors": {
        "url": "https://huggingface.co/datasets/Jwerner00/unmasked-assets/resolve/main/reed_pony_v2.safetensors",
        "auth": f"Bearer {HF_TOKEN_VAL}", "size_gb": 0.14
    },
}

def log(msg):
    print(f"[handler] {msg}", flush=True)

def ensure_models():
    log("Checking models...")
    volume_models = os.path.join(MODEL_VOLUME, "models")
    comfy_models = os.path.join(COMFY_DIR, "models")

    # Diagnostic: what's on the volume?
    log(f"Volume mount exists: {os.path.exists(MODEL_VOLUME)}")
    log(f"Volume models dir exists: {os.path.exists(volume_models)}")
    if os.path.exists(volume_models):
        for root, dirs, files in os.walk(volume_models):
            for f in files:
                fp = os.path.join(root, f)
                sz = os.path.getsize(fp) / (1024*1024)
                log(f"  Volume file: {os.path.relpath(fp, volume_models)} ({sz:.1f}MB)")
    else:
        log("  Volume models dir MISSING — will create and download everything")

    for rel_path, info in MODELS.items():
        vol_path = os.path.join(volume_models, rel_path)
        comfy_path = os.path.join(comfy_models, rel_path)
        os.makedirs(os.path.dirname(vol_path), exist_ok=True)
        os.makedirs(os.path.dirname(comfy_path), exist_ok=True)

        need_download = False
        expected_bytes = int(info['size_gb'] * 1024 * 1024 * 1024)
        min_bytes = int(expected_bytes * 0.99)  # Must be within 1% of expected
        if not os.path.exists(vol_path):
            log(f"{rel_path}: not on volume, need download")
            need_download = True
        else:
            actual_bytes = os.path.getsize(vol_path)
            actual_mb = actual_bytes / (1024*1024)
            expected_mb = expected_bytes / (1024*1024)
            if actual_bytes < min_bytes:
                log(f"{rel_path}: CORRUPT — {actual_mb:.1f}MB on disk, expected {expected_mb:.0f}MB (min {min_bytes/1024/1024:.0f}MB) — deleting and re-downloading")
                os.remove(vol_path)
                need_download = True
            else:
                log(f"{rel_path}: on volume ({actual_mb:.1f}MB, expected {expected_mb:.0f}MB) ✓")

        if need_download:
            log(f"Downloading {rel_path} (~{info['size_gb']}GB)...")
            cmd = ["curl", "-fSL", "--retry", "3", "--retry-delay", "5",
                   "-o", vol_path, "-A", "Mozilla/5.0 (ComfyUI-Worker)"]
            if info.get("auth"):
                cmd += ["-H", f"Authorization: {info['auth']}"]
            cmd.append(info["url"])
            result = subprocess.run(cmd, timeout=1800, capture_output=True, text=True)
            if result.returncode != 0:
                log(f"wget FAILED for {rel_path} (exit {result.returncode})")
                log(f"  stderr: {result.stderr[:500]}")
                # Delete partial/corrupt file so next boot retries cleanly
                if os.path.exists(vol_path):
                    sz = os.path.getsize(vol_path)
                    log(f"  deleting partial file ({sz} bytes) to prevent corruption on next boot")
                    os.remove(vol_path)
                continue
            # Verify download size — delete if too small (error page / partial)
            if os.path.exists(vol_path):
                sz = os.path.getsize(vol_path) / (1024*1024)
                expected_mb = info['size_gb'] * 1024
                log(f"Downloaded: {rel_path} ({sz:.1f}MB, expected ~{expected_mb:.0f}MB)")
                if sz < 10:
                    log(f"  FAILED: file only {sz:.1f}MB — deleting (likely error page or partial)")
                    os.remove(vol_path)
                    continue
            else:
                log(f"  FAILED: file not found after curl completed")
                continue

        # Symlink into ComfyUI
        if os.path.islink(comfy_path):
            target = os.readlink(comfy_path)
            log(f"Symlink exists: {rel_path} -> {target}")
        elif not os.path.exists(comfy_path):
            os.symlink(vol_path, comfy_path)
            log(f"Symlinked: {rel_path} -> {vol_path}")

    # Diagnostic: what does ComfyUI see?
    ckpt_dir = os.path.join(comfy_models, "checkpoints")
    lora_dir = os.path.join(comfy_models, "loras")
    log(f"ComfyUI checkpoints dir: {os.listdir(ckpt_dir) if os.path.exists(ckpt_dir) else 'MISSING'}")
    log(f"ComfyUI loras dir: {os.listdir(lora_dir) if os.path.exists(lora_dir) else 'MISSING'}")
    log("Model check complete")

def start_comfyui():
    log("Starting ComfyUI...")
    comfy_log = "/workspace/comfyui.log"
    # Write extra_model_paths.yaml to point ComfyUI directly at volume models
    extra_paths_file = os.path.join(COMFY_DIR, "extra_model_paths.yaml")
    extra_paths_content = f"""runpod_volume:
    base_path: {MODEL_VOLUME}/models
    checkpoints: checkpoints
    loras: loras
    ipadapter: ipadapter
    clip_vision: clip_vision
    controlnet: controlnet
"""
    with open(extra_paths_file, "w") as f:
        f.write(extra_paths_content)
    log(f"Wrote extra_model_paths.yaml pointing to {MODEL_VOLUME}/models")

    proc = subprocess.Popen(
        [sys.executable, "main.py", "--listen", "0.0.0.0", "--port", "8188", "--disable-xformers"],
        cwd=COMFY_DIR,
        stdout=open(comfy_log, "w"),
        stderr=subprocess.STDOUT
    )
    for i in range(300):
        try:
            urllib.request.urlopen(f"{COMFY_HOST}/system_stats", timeout=5)
            log(f"ComfyUI ready after {i+1}s")
            return proc
        except:
            time.sleep(1)
        if proc.poll() is not None:
            tail = ""
            try:
                with open(comfy_log) as f:
                    tail = f.read()[-2000:]
            except: pass
            raise RuntimeError(f"ComfyUI exited (code {proc.returncode}). Log:\n{tail}")
    tail = ""
    try:
        with open(comfy_log) as f:
            tail = f.read()[-2000:]
    except: pass
    proc.kill()
    raise RuntimeError(f"ComfyUI timeout after 300s. Log:\n{tail}")

def queue_workflow(workflow, client_id="serverless"):
    data = json.dumps({"prompt": workflow, "client_id": client_id}).encode()
    req = urllib.request.Request(f"{COMFY_HOST}/prompt", data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode('utf-8', errors='replace')
        raise RuntimeError(f"ComfyUI /prompt rejected ({e.code}): {body}")
    if result.get("node_errors"):
        raise ValueError(f"Node errors: {json.dumps(result['node_errors'])}")
    return result["prompt_id"]

def wait_for_result(prompt_id, timeout=600):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(f"{COMFY_HOST}/history/{prompt_id}", timeout=10) as resp:
                history = json.loads(resp.read())
            if prompt_id in history:
                entry = history[prompt_id]
                status = entry.get("status", {})
                status_str = status.get("status_str", "")

                if status_str == "error":
                    raise RuntimeError(f"Generation error: {status.get('messages')}")

                # Look for output images
                outputs = entry.get("outputs", {})
                for node_id, output in outputs.items():
                    for img in output.get("images", []):
                        return img

                # Prompt is in history but no images found
                if status_str == "success" or outputs:
                    # Completed but produced nothing — don't poll forever
                    log(f"Prompt {prompt_id} finished (status={status_str}) but no images in outputs: {list(outputs.keys())}")
                    raise RuntimeError(f"Generation completed but produced no images. Status: {status_str}, outputs: {json.dumps({k: list(v.keys()) for k,v in outputs.items()})}")
        except urllib.error.URLError:
            pass
        time.sleep(2)

        # Log progress every 30s
        elapsed = int(time.time() - start)
        if elapsed % 30 < 3:
            try:
                with urllib.request.urlopen(f"{COMFY_HOST}/queue", timeout=5) as resp:
                    queue = json.loads(resp.read())
                running = len(queue.get("queue_running", []))
                pending = len(queue.get("queue_pending", []))
                log(f"  [{elapsed}s] Queue: {running} running, {pending} pending")
            except:
                log(f"  [{elapsed}s] (queue check failed)")

    raise TimeoutError(f"Generation timed out after {timeout}s")

def download_image(filename):
    url = f"{COMFY_HOST}/view?filename={urllib.parse.quote(filename)}&subfolder=&type=output"
    with urllib.request.urlopen(url, timeout=30) as resp:
        return resp.read()

_comfyui_proc = None
_initialized = False
_init_error = None
_init_thread = None

def _run_initialize():
    global _comfyui_proc, _initialized, _init_error
    try:
        log("Background init: downloading models...")
        ensure_models()
        log("Background init: starting ComfyUI...")
        _comfyui_proc = start_comfyui()
        _initialized = True
        log("Worker initialized successfully")
    except Exception as e:
        _init_error = str(e)
        log(f"INIT ERROR: {e}")

def save_input_images(images):
    """Save base64-encoded images to ComfyUI's input directory so LoadImage can reference them."""
    input_dir = os.path.join(COMFY_DIR, "input")
    os.makedirs(input_dir, exist_ok=True)
    for img in images:
        name = img.get("name", "input.png")
        data = base64.b64decode(img["data"])
        path = os.path.join(input_dir, name)
        with open(path, "wb") as f:
            f.write(data)
        log(f"Saved input image: {name} ({len(data)} bytes)")

def handler(job):
    if _init_thread and _init_thread.is_alive():
        log("Waiting for initialization...")
        _init_thread.join(timeout=600)
    if not _initialized:
        return {"error": f"Worker not initialized: {_init_error or 'unknown'}"}
    job_input = job["input"]
    workflow = job_input.get("workflow")
    client_id = job_input.get("client_id", f"job-{job.get('id', 'unknown')}")
    if not workflow:
        return {"error": "No workflow provided"}
    # Save any input images to ComfyUI's input directory before running workflow
    if "images" in job_input:
        save_input_images(job_input["images"])
    log(f"Processing job {client_id}")
    prompt_id = queue_workflow(workflow, client_id)
    log(f"Queued: {prompt_id}")
    img_info = wait_for_result(prompt_id)
    log(f"Generated: {img_info['filename']}")
    img_bytes = download_image(img_info["filename"])
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    return {"image_b64": img_b64, "filename": img_info["filename"], "prompt_id": prompt_id}

_init_thread = threading.Thread(target=_run_initialize, daemon=True)
_init_thread.start()
log("Background init thread started")

try:
    runpod.serverless.start({"handler": handler})
except Exception as e:
    log(f"FATAL: {e}")
    raise