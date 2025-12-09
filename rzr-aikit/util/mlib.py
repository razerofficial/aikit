from pynvml import * # functions are prefixed so no namespace pollution
from prometheus_client.parser import text_string_to_metric_families
import logging
import urllib.request
from urllib.error import URLError

logger = logging.getLogger(__name__)

def get_cuda_total_vram(distributed: bool = False) -> float:
    """
    Get total VRAM that can be used for loading a model
    """
    from gpu_discovery.discover import discover_gpus

    all_gpus = []
    result = discover_gpus()

    if distributed:
        if "FAIL" in result:
            return -1.0
        
        try:
            nodes = result["ray"]
            for node in nodes:
                all_gpus.extend(node["gpus"])
        except (KeyError, TypeError):
            return 0.0
    else:
        try:
            gpus = result["local"]
            all_gpus.extend(gpus)
        except Exception:
            nvmlInit()
            device_count = nvmlDeviceGetCount()
            handles = [nvmlDeviceGetHandleByIndex(i) for i in range(device_count)]
            mem_infos = [nvmlDeviceGetMemoryInfo(handle) for handle in handles]
            return sum(info.total for info in mem_infos)
        
    total_memory = sum(
        gpu["total_memory"] for gpu in all_gpus if gpu["total_memory"] is not None
    )
    return total_memory

def get_cuda_gpu_infos():
    nvmlInit() # almost instant
    def get_info_from_handle(handle):
        return {
            "name": nvmlDeviceGetName(handle),
            "uuid": nvmlDeviceGetUUID(handle),
            # "serial": nvmlDeviceGetSerial(handle),
            "mem": nvmlDeviceGetMemoryInfo(handle),
        }
    device_count = nvmlDeviceGetCount()
    handles = [nvmlDeviceGetHandleByIndex(i) for i in range(device_count)]
    return map(get_info_from_handle, handles)

def get_metrics():
    try:
        with urllib.request.urlopen("http://localhost:8000/metrics") as u:
            metrics = u.read().decode('utf-8')
        return list(text_string_to_metric_families(metrics))
    except URLError:
        return []
    
def check_model_fit(memory: float, weight_size: int, dtype: str) -> str:
    import re

    try:
        extracted_dtype = int(re.search(r"\d+", dtype).group())
    except (AttributeError, ValueError):
        print(f"Warning: Unable to extract bits from dtype '{dtype}'. Assuming 16-bit.")
        extracted_dtype = 16
        
    if weight_size * 1.2 < memory:
        fit = "✅ Yes"
    elif weight_size * (4 / extracted_dtype) * 1.2 < memory:
        fit = "🟡 Limited"
    else:
        fit = "❌ No"
    return fit


from openai import OpenAI, APIConnectionError
client = OpenAI(base_url="http://localhost:8000/v1", api_key="aaa")

def get_running_models():
    try:
        return client.models.list()
    except APIConnectionError:
        return []

def comp_generate(*args, **kwargs):
    return client.completions.create(*args, **kwargs)
