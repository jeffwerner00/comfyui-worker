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
        "url": "https://civitai.com/api/download/models/2155386",
        "auth": f"Bearer {CIVITAI_TOKEN}", "size_gb": 6.9
    },
    "loras/riley-pony-v1.safetensors": {
        "url": "https://huggingface.co/datasets/Jwerner00/unmasked-assets/resolve/main/riley-pony-v1.safetensors",
        "auth": f"Bearer {HF_TOKEN_VAL}", "size_gb": 0.14
    },
    "loras/pam-pony-v1.safetensors": {
        "url": "https://huggingface.co/datasets/Jwerner00/unmasked-assets/resolve/main/pam-pony-v1.safetensors",
        "auth": f"Bearer {HF_TOKEN_VAL}", "size_gb": 0.14
    },
    "loras/vera-pony-v1.safetensors": {
        "url": "https://huggingface.co/datasets/Jwerner00/unmasked-assets/resolve/main/vera-pony-v1.safetensors",
        "auth": f"Bearer {HF_TOKEN_VAL}", "size_gb": 0.14
    },
    "loras/nova-pony-v1.safetensors": {
        "url": "https://huggingface.co/datasets/Jwerner00/unmasked-assets/resolve/main/nova-pony-v1.safetensors",
        "auth": f"Bearer {HF_TOKEN_VAL}", "size_gb": 0.14
    },
}

def log(msg):
    print(f"[handler] {msg}", flush=True)

def ensure_models():
    log("Checking models...")
    volume_models = os.path.join(MODEL_VOLUME, "models")
    comfy_models = os.path.join(COMFY_DIR, "models")
    for rel_path, info in MODELS.items():
        vol_path = os.path.join(volume_models, rel_path)
        comfy_path = os.path.join(comfy_models, rel_path)
        os.makedirs(os.path.dirname(vol_path), exist_ok=True)
        os.makedirs(os.path.dirname(comfy_path), exist_ok=True)
        if not os.path.exists(vol_path) or os.path.getsize(vol_path) < 1_000_000:
            log(f"Downloading {rel_path} (~{info['size_gb']}GB)...")
            cmd = ["wget", "-q", "-L", "--show-progress", "-O", vol_path]
            if info.get("auth"):
                cmd += ["--header", f"Authorization: {info['auth']}"]
            cmd.append(info["url"])
            result = subprocess.run(cmd, timeout=1800)
            if result.returncode != 0:
                log(f"wget failed for {rel_path}")
                continue
            log(f"Downloaded: {rel_path}")
        if not os.path.exists(comfy_path):
            os.symlink(vol_path, comfy_path)
            log(f"Symlinked: {rel_path}")
    log("Model check complete")

def start_comfyui():
    log("Starting ComfyUI...")
    comfy_log = "/workspace/comfyui.log"
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

def wait_for_result(prompt_id, timeout=300):
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