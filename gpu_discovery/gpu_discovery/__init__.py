"""
gpu_discovery package

Public API:
- Fp8GemmVerifier: class for FP8 GEMM verification
- Fp4Sm120aGemmVerifier: class for FP4 GEMM verification
- collect_nvml_info(): function to collect nvml info
- refresh_nvml_info(): function to refresh nvml info
- discover_local_gpus(): function to list local GPUs with FP8/FP4 GFLOPs
- discover_cluster_gpus(): function to list cluster GPUs with FP8/FP4 GFLOPs
- discover_gpus(): function to list local and cluster GPUs with FP8/FP4 GFLOPs
"""

import torch  # ensure torch libs are loaded before extension
from gpu_discovery_cpp import Fp8GemmVerifier, Fp4Sm120aGemmVerifier
from .discover_local_gpus import discover_local_gpus, collect_nvml_info, refresh_nvml_info
from .discover import discover_gpus

__version__ = "0.1.0"

__all__ = [
    "Fp8GemmVerifier",
    "Fp4Sm120aGemmVerifier",
    "collect_nvml_info",
    "refresh_nvml_info",
    "discover_local_gpus",
    "discover_gpus",
    "__version__",
]

# Try importing Ray-dependent cluster discovery
try:
    from .discover_cluster_gpus import discover_cluster_gpus
    __all__.append("discover_cluster_gpus")
except ImportError:
    discover_cluster_gpus = None
