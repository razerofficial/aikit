import pynvml
import sys
from rzr_aikit.utils.mlib import get_device_memory_info_with_fallback

# Global cache for NVML static info (bus id, uuid, total mem, etc.)
_GPU_NVML_INFO = None


def collect_nvml_info():
    """Collect static GPU info using NVML (including free memory snapshot)."""
    pynvml.nvmlInit()
    num_gpus = pynvml.nvmlDeviceGetCount()
    gpu_nvml_info = []

    for i in range(num_gpus):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        uuid = str(pynvml.nvmlDeviceGetUUID(handle))
        bus_id = pynvml.nvmlDeviceGetPciInfo(handle).busId
        mem_info = get_device_memory_info_with_fallback(handle)

        gpu_nvml_info.append({
            "device_id": i,
            "uuid": uuid,
            "bus_id": bus_id,
            "total_memory": mem_info.total,
            "free_memory": mem_info.free,  # snapshot at collection
        })

    pynvml.nvmlShutdown()
    return gpu_nvml_info


def refresh_nvml_info():
    global _GPU_NVML_INFO
    _GPU_NVML_INFO = collect_nvml_info()


def get_nvml_info_by_uuid(uuid: str):
    """
    Return GPU info dict from _GPU_NVML_INFO matching a given UUID.
    If not found, return None.
    """
    global _GPU_NVML_INFO
    if _GPU_NVML_INFO is None:
        return None
    for gpu in _GPU_NVML_INFO:
        if gpu["uuid"] == uuid:
            return gpu
    return None


def _warmup_large_gemm(device_id, iters=300, size=8192):
    """Run large GEMMs to force GPU clocks to max boost."""
    import torch
    torch.cuda.set_device(device_id)
    A = torch.randn(size, size, device=f"cuda:{device_id}", dtype=torch.float32)
    B = torch.randn(size, size, device=f"cuda:{device_id}", dtype=torch.float32)
    for _ in range(iters):
        _ = torch.matmul(A, B)
    torch.cuda.synchronize()


def discover_local_gpus(verbose=False):
    """Discover local GPUs and measure FP8/FP4 GFLOPs."""
    # Step 1: Load cache
    from .cache import load_cache
    cache_gpus_info = load_cache()
    if verbose:
        import json
        print("cache_gpus_info :")
        print(json.dumps(cache_gpus_info, indent=2))
        print()

    global _GPU_NVML_INFO

    # Step 2: Initialize NVML info once
    if _GPU_NVML_INFO is None:
        _GPU_NVML_INFO = collect_nvml_info()
        if verbose:
            print("_GPU_NVML_INFO is collected")
    else:
        if verbose:
            print("_GPU_NVML_INFO already set")

    if verbose:
        import json
        print(json.dumps(_GPU_NVML_INFO, indent=2))
        print()

    # Step 3: Torch-based benchmarks
    import torch
    from gpu_discovery_cpp import Fp8GemmVerifier, Fp4Sm120aGemmVerifier

    results = []
    verifier_fp8 = Fp8GemmVerifier()
    verifier_fp4 = Fp4Sm120aGemmVerifier()

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)

        uuid = "GPU-" + str(props.uuid)

        # Locate _GPU_NVML_INFO based on UUID
        info = get_nvml_info_by_uuid(uuid)

        gpu_info = {
            "device_id": i,
            "name": props.name,
            "uuid": uuid,
            "bus_id": info.get("bus_id", "None") if info else "None",
            "compute_capability": f"{props.major}.{props.minor}",
            "total_memory": info.get("total_memory", "None") if info else "None",
            "free_memory":  info.get("free_memory",  "None") if info else "None",
            "multi_processor_count": props.multi_processor_count,
            "fp8_support": False,
            "fp8_gflops": 0.0,
            "fp4_support": False,
            "fp4_gflops": 0.0,
        }

        # check cache GPUs info
        cache_info = None
        for thisGPU in cache_gpus_info:
            if thisGPU.get("uuid") == uuid:
                cache_info = thisGPU
                break;

        if cache_info:
            gpu_info["fp8_support"] = cache_info["fp8_support"]
            gpu_info["fp8_gflops"] = cache_info["fp8_gflops"]

            gpu_info["fp4_support"] = cache_info["fp4_support"]
            gpu_info["fp4_gflops"] = cache_info["fp4_gflops"]
        else:
            # FP8 benchmark
            try:
                _warmup_large_gemm(i)
                ok, error_code, gflops = verifier_fp8.verify(device_id=i, verbose=verbose)
                gpu_info["fp8_support"] = bool(ok)
                gpu_info["fp8_gflops"] = gflops if ok else 0.0
            except Exception:
                pass

            # FP4 benchmark
            try:
                _warmup_large_gemm(i)
                ok, error_code, gflops = verifier_fp4.verify(device_id=i, verbose=verbose)
                gpu_info["fp4_support"] = bool(ok)
                gpu_info["fp4_gflops"] = gflops if ok else 0.0
            except Exception:
                pass

        results.append(gpu_info)

    return results
