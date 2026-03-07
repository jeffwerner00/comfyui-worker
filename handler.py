"""Minimal test handler - confirms image boots and RunPod SDK works"""
import runpod
import threading
import time

def log(msg):
    print(f"[handler] {msg}", flush=True)

def handler(job):
    log(f"Job received: {job.get('id')}")
    return {"status": "ok", "message": "minimal handler working"}

log("Starting minimal test handler...")
log("Registering with RunPod...")

try:
    runpod.serverless.start({"handler": handler})
except Exception as e:
    log(f"FATAL: {e}")
    raise