import os
import json

CACHE_FILE = os.path.expanduser("~/.cache/huggingface/gpu_discovery.json")

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return []

def save_cache(data):
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(data, f, indent=2)

def find_gpu_in_cache(uuid):
    """Return cached GPU info by UUID, or None if not found."""
    cache = load_cache()
    for gpu in cache:
        if gpu.get("uuid") == uuid:
            return gpu
    return None

def delete_cache():
    """Delete the GPU discovery cache file if it exists."""
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)
        return True
    return False
