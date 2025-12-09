#!/usr/bin/env python3
"""
GPU Selection Tool for AI Model Inference

This script analyzes available GPUs and selects the optimal configuration for running
large language models with VLLM. It considers memory requirements, performance characteristics,
and distributed inference constraints to recommend the best GPU allocation strategy.

Key Features:
- Analyzes model size, quantization, and layer count from HuggingFace
- Discovers compatible GPUs based on tensor format support
- Applies intelligent memory overhead calculations (20% buffer)
- Handles both local and distributed GPU configurations
- Generates ready-to-run commands with proper environment variables

GPU Selection Strategies (tried in order):
1. Single-Node Flexible: Tries to fit the model on GPUs from a single machine using
   memory pooling. Prefers fewer GPUs and single-node deployment for simplicity.

2. Multi-GPU Pooled: Uses multiple GPUs across potentially multiple nodes by pooling
   their memory. Minimizes GPU count first, then node count, then maximizes performance.

Distributed Mode Requirements:
- When distributed GPUs are detected, at least one local GPU must be included
- The local GPU serves as the coordinator node for distributed inference
- Uses Ray backend for multi-node coordination

Output:
- GPU filtering commands (CUDA_VISIBLE_DEVICES or gpu-filter)
- Pipeline partitioning variables (VLLM_PP_LAYER_PARTITION)
- Ready-to-run rzr-aikit commands with appropriate flags
"""

import argparse
import sys
import json
import itertools
import math
from dataclasses import dataclass
from typing import Optional

from util.ModelInfoFetcher import ModelInfoFetcher

# Configuration constants
MEMORY_OVERHEAD_FACTOR = 1.2  # 20% overhead for model loading
FIRST_GPU_REDUCTION = 0.25   # 25% reduction for coordination overhead in distributed setups
MIN_GPU_WEIGHT = 0.1         # Minimum weight for GPU in proportional partitioning
BYTES_PER_GB = 1024**3       # Conversion factor
DEFAULT_MAX_POOL = 12        # Default maximum GPU combinations to consider
DEFAULT_GPUS_PATH = "./gpus.json"

# Quantization factors for FP32 models
QUANT_FACTORS_FP32 = {
    "mxfp4": 0.140625,
    "nvfp4": 0.140625,
    "fp4": 0.140625,
    "gptq-4bit": 0.14,
    "awq-4bit": 0.1667,
    "nf4": 0.16,
    "int8": 0.25,
    "fp8": 0.25,
    "fp16": 0.5,
    "bf16": 0.5,
    "fp32": 1.0,
}

# Tensor format mappings
FORMAT_TOKENS = {
    "float64": ["FP64"],
    "float32": ["FP32"],
    "float16": ["FP16", "BF16"],
    "bfloat16": ["BF16", "FP16"],
    "float8": ["FP8_E4M3", "FP8_E5M2"],
    "float6": ["FP6_E2M3", "FP6_E3M2"],
    "float4": ["FP4_E2M1"],
    "int32": ["INT32"],
    "int16": ["INT16"],
    "int8": ["INT8"],
    "int4": ["INT4"],
}


@dataclass
class GPUInfo:
    """GPU information dataclass matching the JSON format."""

    idx: int
    name: str
    ip: str
    pci_bus_id: Optional[str]
    uuid: str
    compute_capability: Optional[float]
    memory: Optional[dict]  # {"total": int, "free": int, "used": int}
    gflops: Optional[float]
    tensor_core_formats: list[str]

    @classmethod
    def from_dict(cls, data: dict) -> "GPUInfo":
        """Create GPUInfo from dictionary."""
        return cls(
            idx=data.get("idx", 0),
            name=data.get("name", ""),
            ip=data.get("ip", ""),
            pci_bus_id=data.get("pci_bus_id"),
            uuid=data.get("uuid", ""),
            compute_capability=data.get("compute_capability"),
            memory=data.get("memory"),
            gflops=data.get("gflops"),
            tensor_core_formats=data.get("tensor_core_formats", []),
        )

    def get_memory_bytes(self) -> tuple[Optional[int], Optional[int]]:
        """Get total and free memory in bytes (raw values)."""
        if not self.memory or not isinstance(self.memory, dict):
            return None, None

        total = self.memory.get("total")
        free = self.memory.get("free")

        if total is None or free is None:
            return None, None

        return int(total), int(free)

    def get_tensor_formats_set(self) -> set[str]:
        """Get tensor formats as a set."""
        return set(self.tensor_core_formats) if self.tensor_core_formats else set()

    def get_gflops(self) -> float:
        """Get GFLOPS value, defaulting to 0.0 if None."""
        return self.gflops if self.gflops is not None else 0.0


class ModelAnalyzer:
    """Handles model information retrieval and analysis."""
    
    def __init__(self, model_id: str):
        self.model_id = model_id
        
        # Check connectivity to respect offline mode
        from util.connectivity import check_huggingface_connectivity
        has_connection = check_huggingface_connectivity()
        
        # Try to initialize fetcher, handling gated models
        try:
            self.fetcher = ModelInfoFetcher(model_id, allow_internet=has_connection)
        except Exception as e:
            # Check if it's a gated model requiring authentication
            if "HuggingfaceAccessTokenRequired" in str(type(e).__name__):
                from util.HuggingfaceHubToken import HuggingfaceHubToken
                
                try:
                    hug = HuggingfaceHubToken()
                    token = hug.get_access_token()
                    self.fetcher = ModelInfoFetcher(
                        model_id, access_token=token, allow_internet=has_connection
                    )
                except Exception:
                    raise
            else:
                raise
    
    def get_quantization_info(self) -> tuple[str, Optional[str]]:
        """Get datatype and quantization method if model is quantized."""
        try:
            dtype = self.fetcher.dtype
            quant_config = self.fetcher.quant_config
            return dtype, quant_config.get("quant_method") if quant_config else None
        except Exception as e:
            raise ValueError("Error detecting quantization method") from e
    
    def get_weights_size_gb(self) -> Optional[float]:
        """Get model weights size in GiB (binary GB, 1024^3)."""
        try:
            # Use ModelInfoFetcher's total_bytes directly for consistency
            total_bytes = self.fetcher.total_bytes
            
            if total_bytes and total_bytes > 0:
                return total_bytes / BYTES_PER_GB
            
            # Fallback: if total_bytes is None, return None
            return None
        except Exception as e:
            raise ValueError("Error fetching model size") from e
    
    def get_num_layers(self) -> Optional[int]:
        """Get number of layers from model config."""
        try:
            config = self.fetcher.config
            if not config:
                return None
            
            # Try common layer count keys
            for key in ("num_hidden_layers", "n_layer", "num_layers"):
                if isinstance(config.get(key), int):
                    return int(config[key])
            
            # Handle encoder-decoder models
            enc = next((config[k] for k in ("encoder_layers", "num_encoder_layers") 
                       if isinstance(config.get(k), int)), None)
            dec = next((config[k] for k in ("decoder_layers", "num_decoder_layers") 
                       if isinstance(config.get(k), int)), None)
            
            if enc is not None and dec is not None:
                return int(enc) + int(dec)
        except Exception:
            pass
        return None


def proportional_partition(total_layers: Optional[int], weights: list[float]) -> Optional[list[int]]:
    """Partition layers proportionally based on weights (VRAM), with first GPU overhead adjustment."""
    if total_layers is None:
        return None
    
    if len(weights) <= 1:
        return [total_layers] if weights else []
    
    # Apply first GPU overhead reduction for coordination overhead
    adjusted_weights = weights.copy()
    adjusted_weights[0] = max(weights[0] * (1 - FIRST_GPU_REDUCTION), MIN_GPU_WEIGHT)
    
    # Calculate proportional allocation based on adjusted weights
    s = sum(max(w, 0.0) for w in adjusted_weights) or 1.0
    raw = [total_layers * (max(w, 0.0) / s) for w in adjusted_weights]
    base = [int(math.floor(x)) for x in raw]
    rem = total_layers - sum(base)
    
    # Distribute remaining layers, but prefer non-first GPUs
    order = sorted(range(len(raw)), key=lambda i: (i == 0, -(raw[i] - base[i])))
    for i in range(rem):
        base[order[i]] += 1
    
    return base


def is_local_gpu(gpu: GPUInfo) -> bool:
    """Check if a GPU is local (marked with [local] in name or has no valid IP)."""
    # Check for explicit [local] marker in name
    if "[local]" in gpu.name:
        return True
    
    # If no explicit marker, check if IP is missing/invalid (indicates local GPU)
    return gpu.ip in (None, "", "N/A", "localhost", "127.0.0.1")


def has_distributed_gpus(candidates: list[GPUInfo]) -> bool:
    """Check if any GPUs in the list are distributed (have valid IP addresses)."""
    return any(not is_local_gpu(gpu) for gpu in candidates)


def filter_with_local_requirement(candidates: list[GPUInfo], selected: list[GPUInfo], 
                                model_quant_gb: Optional[float] = None) -> Optional[list[GPUInfo]]:
    """
    If there are distributed GPUs in candidates, ensure at least one local GPU is in selected.
    Returns the filtered selection or None if requirement cannot be met.
    """
    if not has_distributed_gpus(candidates):
        # No distributed GPUs in candidates, no local requirement needed
        return selected
    
    # There are distributed GPUs, check if we have at least one local GPU in selection
    local_in_selected = [gpu for gpu in selected if is_local_gpu(gpu)]
    
    if local_in_selected:
        # Already have at least one local GPU, requirement satisfied
        return selected
    
    # No local GPUs in selection, need to add one
    local_candidates = [gpu for gpu in candidates if is_local_gpu(gpu)]
    
    if not local_candidates:
        # No local GPUs available at all, cannot satisfy requirement
        return None
    
    # If we have memory requirements, filter local candidates by those requirements first
    if model_quant_gb is not None:
        per_need_gb = (model_quant_gb / len(selected)) * MEMORY_OVERHEAD_FACTOR
        per_need_bytes = int(per_need_gb * BYTES_PER_GB)
        
        # Filter local candidates that can meet the memory requirement
        viable_local_candidates = []
        for gpu in local_candidates:
            _, free_bytes = gpu.get_memory_bytes()
            if free_bytes is not None and free_bytes >= per_need_bytes:
                viable_local_candidates.append(gpu)
        
        if not viable_local_candidates:
            # No local GPUs can meet the memory requirements
            return None
        
        local_candidates = viable_local_candidates
    
    # Replace the least performant GPU in selection with the best local GPU
    if len(selected) > 1:
        # Sort selected by performance (worst first)
        selected_sorted = sorted(selected, key=lambda x: (x.get_gflops(), x.get_memory_bytes()[1] or 0))
        # Sort local candidates by performance (best first)
        local_sorted = sorted(local_candidates, key=lambda x: (x.get_gflops(), x.get_memory_bytes()[1] or 0), reverse=True)
        
        # Replace worst selected with best local
        new_selected = selected_sorted[1:] + [local_sorted[0]]
        return new_selected
    else:
        # Single GPU case - replace with best local GPU that can handle the model
        if model_quant_gb is not None:
            single_need_gb = model_quant_gb * MEMORY_OVERHEAD_FACTOR
            single_need_bytes = int(single_need_gb * BYTES_PER_GB)
            
            viable_singles = []
            for gpu in local_candidates:
                _, free_bytes = gpu.get_memory_bytes()
                if free_bytes is not None and free_bytes >= single_need_bytes:
                    viable_singles.append(gpu)
            
            if not viable_singles:
                return None
            
            local_sorted = sorted(viable_singles, key=lambda x: (x.get_gflops(), x.get_memory_bytes()[1] or 0), reverse=True)
        else:
            local_sorted = sorted(local_candidates, key=lambda x: (x.get_gflops(), x.get_memory_bytes()[1] or 0), reverse=True)
        
        return [local_sorted[0]]


def try_single_node_first(
    candidates: list[GPUInfo], model_quant_gb: float, verbose: bool = False
) -> Optional[list[GPUInfo]]:
    """Strategy 1: single-node flexible partitioning (fit model using memory pooling on same machine)."""
    if verbose:
        print("\n[info] Strategy 1: single-node flexible partitioning (fit model using memory pooling on same machine)")

    grouped: dict[str, list[GPUInfo]] = {}
    for g in candidates:
        grouped.setdefault(g.ip, []).append(g)

    for ip, gpus in grouped.items():
        gpus.sort(
            key=lambda x: (x.get_gflops(), x.get_memory_bytes()[1] or 0), reverse=True
        )
        
        # Try k = 1..N GPUs on this node with flexible partitioning
        for k in range(1, len(gpus) + 1):
            # Try all combinations of k GPUs from this node
            from itertools import combinations
            for gpu_combo in combinations(gpus, k):
                # Calculate total available memory
                total_free_bytes = 0
                for gpu in gpu_combo:
                    _, free_bytes = gpu.get_memory_bytes()
                    if free_bytes is not None:
                        total_free_bytes += free_bytes
                    else:
                        total_free_bytes = 0
                        break
                
                # Check if we have enough total memory for the model (with overhead)
                needed_bytes = int(model_quant_gb * MEMORY_OVERHEAD_FACTOR * BYTES_PER_GB)
                
                if verbose:
                    combo_indices = [g.idx for g in gpu_combo]
                    total_free_gb = total_free_bytes / BYTES_PER_GB
                    needed_gb = needed_bytes / BYTES_PER_GB
                    print(f"[debug] node {ip}: k={k}, combo={combo_indices}, total_free={total_free_gb:.2f}GB, needed={needed_gb:.2f}GB")
                
                if total_free_bytes >= needed_bytes:
                    # This combination works! Apply local requirement filter
                    success = list(gpu_combo)
                    filtered_success = filter_with_local_requirement(candidates, success, model_quant_gb)
                    if filtered_success is not None:
                        if verbose:
                            print(f"[info] single-node success on {ip}: k={k} -> GPUs {[x.idx for x in filtered_success]}")
                        return filtered_success
                    elif verbose and has_distributed_gpus(candidates):
                        print(f"[debug] node {ip}: combo rejected - distributed mode requires at least one local GPU")

        if verbose:
            print(f"[warn] node {ip} cannot host model with any GPU combination≤{len(gpus)}")
    return None





def try_pooled(
    candidates: list[GPUInfo],
    model_quant_gb: float,
    max_pool: int,
    verbose: bool = False,
) -> Optional[list[GPUInfo]]:
    """Strategy 2: multi-GPU pooled memory (aggregate memory across GPUs, minimize GPU count)."""
    if verbose:
        print("\n[info] Strategy 2: multi-GPU pooled memory (aggregate memory across GPUs)")

    total_need_gb = model_quant_gb * MEMORY_OVERHEAD_FACTOR
    total_need_bytes = int(total_need_gb * BYTES_PER_GB)
    pool = candidates[: max(1, min(max_pool, len(candidates)))]

    # Try from smallest to largest number of GPUs to minimize GPU count
    for k in range(1, len(pool) + 1):
        viable_combos = []
        
        # Find all viable combinations for this k
        for combo in itertools.combinations(pool, k):
            total_free_bytes = 0
            for gpu in combo:
                _, free_bytes = gpu.get_memory_bytes()
                if free_bytes is not None:
                    total_free_bytes += free_bytes
                else:
                    total_free_bytes = 0
                    break

            if total_free_bytes >= total_need_bytes:
                ips = {x.ip for x in combo}
                # Score: prefer fewer nodes first, then higher total GFLOPS
                score = (-len(ips), sum(x.get_gflops() for x in combo))
                viable_combos.append((combo, score))
        
        if not viable_combos:
            if verbose:
                print(f"[debug] pooled k={k}, found=no")
            continue
            
        # Sort viable combinations by score (best first)
        viable_combos.sort(key=lambda x: x[1], reverse=True)
        
        if verbose:
            print(f"[debug] pooled k={k}, found={len(viable_combos)} combinations")
        
        # Try each combination in order until we find one that passes the local filter
        for combo, score in viable_combos:
            best = list(combo)
            if verbose:
                ips = {x.ip for x in combo}
                print(f"[debug] pooled k={k}, trying combo: [{', '.join(str(g.idx) for g in best)}], nodes={len(ips)}, score={score}")
            
            # Apply local requirement filter
            filtered_best = filter_with_local_requirement(candidates, best, model_quant_gb)
            if filtered_best is not None:
                if verbose:
                    ips = {x.ip for x in filtered_best}
                    print(
                        f"[info] pooled success: k={k}, nodes={len(ips)}, GPUs={[x.idx for x in filtered_best]}, total_need={total_need_gb:.2f} GiB"
                    )
                return filtered_best  # Return immediately - we found the minimal solution
            else:
                if verbose:
                    print(f"[debug] pooled k={k}, combo rejected - distributed mode requires at least one local GPU")

    if verbose:
        print("[warn] pooled failed for all k")
    return None


def calculate_memory_requirement(model_quant_gb: float, num_gpus: int) -> tuple[float, int]:
    """Calculate per-GPU memory requirement in GB and bytes."""
    per_need_gb = (model_quant_gb / num_gpus) * MEMORY_OVERHEAD_FACTOR
    per_need_bytes = int(per_need_gb * BYTES_PER_GB)
    return per_need_gb, per_need_bytes


def filter_gpus_by_memory(gpus: list[GPUInfo], required_bytes: int) -> list[GPUInfo]:
    """Filter GPUs that have sufficient free memory."""
    return [gpu for gpu in gpus if gpu.get_memory_bytes()[1] and 
            gpu.get_memory_bytes()[1] >= required_bytes]


def find_optimal_gpus(
    candidates: list[GPUInfo],
    model_quant_gb: float,
    max_pool: int,
    verbose: bool = False,
) -> Optional[list[GPUInfo]]:
    """Find optimal GPU combination using strategy ordering: single-node -> pooled."""
    strategies = [
        ("single-node flexible partitioning", try_single_node_first),
        ("multi-GPU pooled memory", try_pooled)
    ]
    
    for strategy_name, strategy_func in strategies:
        if verbose:
            print(f"\n[info] Trying {strategy_name}...")
        
        if strategy_func == try_pooled:
            selected = strategy_func(candidates, model_quant_gb, max_pool, verbose)
        else:
            selected = strategy_func(candidates, model_quant_gb, verbose)
        
        if selected is not None:
            if verbose:
                print(f"[info] Success with {strategy_name}")
            return selected
    
    return None


def load_gpu_data(gpus_path: str) -> list[dict]:
    """Load GPU data from JSON file."""
    try:
        with open(gpus_path, "r") as f:
            gpu_list = json.load(f)
        
        if not isinstance(gpu_list, list) or not gpu_list:
            raise ValueError(f"{gpus_path} is empty/invalid")
        
        return gpu_list
    except Exception as e:
        print(f"[error] Failed to read {gpus_path}: {e}", file=sys.stderr)
        print("[info] Please run 'gpu-discover' first to generate the GPU information file.", file=sys.stderr)
        sys.exit(1)


def display_model_info(analyzer: ModelAnalyzer, export_only: bool = False):
    """Display model information unless in export-only mode."""
    if export_only:
        return
    
    from humanize import naturalsize
    
    datatype, quant_method = analyzer.get_quantization_info()
    size_gb = analyzer.get_weights_size_gb()
    num_layers = analyzer.get_num_layers()
    supported_formats = FORMAT_TOKENS.get(datatype, [])
    
    # Convert to bytes for naturalsize (matches model info display)
    size_bytes = int(size_gb * BYTES_PER_GB) if size_gb else 0
    size_display = naturalsize(size_bytes) if size_bytes > 0 else "N/A"
    
    print(analyzer.model_id)
    print("-" * len(analyzer.model_id))
    print(f"  Precision: {datatype.upper()} | Acceptable formats: {', '.join(sorted(supported_formats))}")
    if quant_method:
        print(f"  Quantization detected: {quant_method.upper()}")
    print(f"  Model size: {size_display}")
    print(f"  Overhead policy: {int(MEMORY_OVERHEAD_FACTOR*100-100)}% of placed partition (Accelerate-style)")
    print(f"  Layers: {num_layers if num_layers is not None else 'N/A'}")


def filter_compatible_gpus(gpu_list: list[dict], supported_formats: list[str]) -> list[GPUInfo]:
    """Filter GPUs that are compatible with the model's tensor formats."""
    candidates = []
    for gpu_data in gpu_list:
        gpu = GPUInfo.from_dict(gpu_data)
        total_bytes, free_bytes = gpu.get_memory_bytes()
        formats = gpu.get_tensor_formats_set()

        if total_bytes is None or free_bytes is None:
            continue
        if not (set(supported_formats) & formats):
            continue

        candidates.append(gpu)
    
    return candidates


def apply_gpu_restrictions(candidates: list[GPUInfo], restrict_gpus: Optional[str], 
                         export_only: bool) -> list[GPUInfo]:
    """Apply GPU index restrictions if specified."""
    if not restrict_gpus:
        return candidates
    
    try:
        restricted_indices = [int(x.strip()) for x in restrict_gpus.split(",") if x.strip()]
    except ValueError:
        print("[error] Invalid --restrict-gpus format. Use comma-separated integers, e.g. '0,2,3'", 
              file=sys.stderr)
        sys.exit(1)
    
    # Validate restricted indices exist in discovered GPUs
    available_indices = {gpu.idx for gpu in candidates}
    invalid_indices = [idx for idx in restricted_indices if idx not in available_indices]
    if invalid_indices:
        print(f"[error] Restricted GPU indices {invalid_indices} not found in discovered GPUs {sorted(available_indices)}", 
              file=sys.stderr)
        sys.exit(1)
    
    # Filter candidates to only include restricted GPUs
    original_count = len(candidates)
    candidates = [gpu for gpu in candidates if gpu.idx in restricted_indices]
    
    if not export_only:
        print(f"\n[info] GPU selection restricted to indices: {sorted(restricted_indices)}")
        print(f"[info] Candidates reduced from {original_count} to {len(candidates)} GPUs")
    
    return candidates


def main(args: argparse.Namespace):
    # Get model information
    analyzer = ModelAnalyzer(args.model)
    datatype, quant_method = analyzer.get_quantization_info()
    size_gb = analyzer.get_weights_size_gb()
    num_layers = analyzer.get_num_layers()
    supported_formats = FORMAT_TOKENS.get(datatype, [])
    
    # Load GPU data
    gpu_list = load_gpu_data(args.gpus_path)
    
    # Display model info (unless export-only)
    display_model_info(analyzer, args.export_only)

    # Filter compatible GPUs and apply restrictions
    all_compatible_gpus = filter_compatible_gpus(gpu_list, supported_formats)  # Keep original list
    candidates = apply_gpu_restrictions(all_compatible_gpus, args.restrict_gpus, args.export_only)
    
    if not candidates:
        print("\n[error] No GPUs satisfy required tensor format(s).", file=sys.stderr)
        sys.exit(1)

    # Sort by performance (speed first, then VRAM)
    candidates.sort(key=lambda gpu: (gpu.get_gflops(), gpu.get_memory_bytes()[1] or 0), reverse=True)

    if not args.export_only and args.verbose:
        print("\nCandidate GPUs (sorted by GFLOPS desc):")
        for gpu in candidates:
            _, free_bytes = gpu.get_memory_bytes()
            free_gb = free_bytes / BYTES_PER_GB if free_bytes else 0
            print(f"  [{gpu.idx}] {gpu.name}")

    # Find optimal GPU selection
    selected = find_optimal_gpus(candidates, size_gb, args.max_pool, args.verbose)

    if selected is None:
        total_free_bytes = sum(gpu.get_memory_bytes()[1] for gpu in candidates 
                              if gpu.get_memory_bytes()[1] is not None)
        total_free_gb = total_free_bytes / BYTES_PER_GB
        single_gpu_need = size_gb * MEMORY_OVERHEAD_FACTOR
        
        # Check if failure might be due to local GPU requirement
        has_distributed = has_distributed_gpus(candidates)
        local_gpus = [gpu for gpu in candidates if is_local_gpu(gpu)]
        
        error_msg = f"\n[error] No suitable GPU configuration found. "
        
        if has_distributed and not local_gpus:
            error_msg += (
                f"Distributed mode detected but no local GPUs available. "
                f"In distributed inference, at least one GPU must be running on the machine where the model is launched (coordinator node). "
            )
        elif has_distributed and local_gpus:
            local_total_gb = sum(gpu.get_memory_bytes()[1] for gpu in local_gpus if gpu.get_memory_bytes()[1]) / BYTES_PER_GB
            error_msg += (
                f"Distributed mode requires at least one local GPU with sufficient memory. "
                f"Local GPUs available: {len(local_gpus)}, total local VRAM: {local_total_gb:.2f} GiB. "
            )
        else:
            error_msg += f"Insufficient free VRAM. "
        
        error_msg += (
            f"Model needs {single_gpu_need:.2f} GiB per GPU, "
            f"total available: {total_free_gb:.2f} GiB across {len(candidates)} compatible GPUs."
        )
        
        print(error_msg, file=sys.stderr)
        sys.exit(1)

    # Generate output
    generate_output(selected, all_compatible_gpus, analyzer.model_id, num_layers, args.export_only)


def calculate_min_gpu_memory_utilization(selected: list[GPUInfo]) -> Optional[float]:
    """Calculate the minimum available memory percentage across selected GPUs.
    
    Returns:
        Minimum available memory percentage (0.0 to 1.0), or None if memory info unavailable.
    """
    min_utilization = None
    
    for gpu in selected:
        total_bytes, free_bytes = gpu.get_memory_bytes()
        if total_bytes is not None and free_bytes is not None and total_bytes > 0:
            utilization = free_bytes / total_bytes
            if min_utilization is None or utilization < min_utilization:
                min_utilization = utilization
    
    return min_utilization


def generate_output(selected: list[GPUInfo], all_compatible_gpus: list[GPUInfo], 
                   model_id: str, num_layers: Optional[int], export_only: bool):
    """Generate and display the output commands and GPU selection info."""
    # Sort selected GPUs by index for consistent output
    selected.sort(key=lambda gpu: gpu.idx)
    gpu_indices = ",".join(str(gpu.idx) for gpu in selected)
    
    # Calculate pipeline partitioning
    free_memory_list = [
        gpu.get_memory_bytes()[1] / BYTES_PER_GB if gpu.get_memory_bytes()[1] else 0.0
        for gpu in selected
    ]
    pp_layers = proportional_partition(num_layers, free_memory_list)
    
    # Display selected GPUs info
    if not export_only:
        print("\nSelected GPUs:")
        for gpu in selected:
            _, free_bytes = gpu.get_memory_bytes()
            free_gb = free_bytes / BYTES_PER_GB if free_bytes else 0
            print(f"[{gpu.idx}] {gpu.name} | IP={gpu.ip} | Free={free_gb:.2f} GiB | GFLOPS={gpu.get_gflops():.1f}")
    
    # Generate commands
    print("\nCommands to run inferencing:")
    
    # Check GPU distribution characteristics
    has_distributed = any(not is_local_gpu(gpu) for gpu in selected)
    local_candidates = [gpu for gpu in all_compatible_gpus if is_local_gpu(gpu)]
    selected_local = [gpu for gpu in selected if is_local_gpu(gpu)]
    need_cuda_filter = len(local_candidates) > len(selected_local) and len(selected_local) > 0
    
    # GPU filtering command
    if has_distributed:
        print(f"gpu-filter {gpu_indices}")
    elif len(selected) > 1 or need_cuda_filter:
        print(f"export CUDA_VISIBLE_DEVICES=\"{gpu_indices}\"")
    
    # Pipeline partitioning (multi-GPU only)
    if len(selected) > 1 and pp_layers is not None:
        print(f'export VLLM_PP_LAYER_PARTITION="{",".join(map(str, pp_layers))}"')
    
    # Main run command
    base_cmd = f"rzr-aikit model run {model_id}"
    if len(selected) > 1:
        cmd = f"{base_cmd} --pipeline-parallel-size {len(selected)}"
        if has_distributed:
            cmd += " --distributed-executor-backend ray"
    else:
        cmd = base_cmd
        if has_distributed:
            cmd += " --distributed-executor-backend ray"
    
    # Add --gpu-memory-utilization if any GPU has less than 90% available memory
    min_utilization = calculate_min_gpu_memory_utilization(selected)
    if min_utilization is not None and min_utilization < 0.90:
        # Use min_utilization minus 2% buffer, but ensure it's at least 0.1
        target_utilization = max(0.1, min_utilization - 0.02)
        cmd += f" --gpu-memory-utilization {target_utilization:.2f}"
    
    print(cmd)


def cli():

    parser = argparse.ArgumentParser(
        prog="gpu-select",
        description="Determine optimal GPUs for the specified model",
    )

    parser.add_argument(
        "model",
        help="HuggingFace model identifier (e.g., 'microsoft/DialoGPT-medium')",
    )

    parser.add_argument(
        "--model-size",
        type=float,
        help="Override base FP32 model size in GB if HuggingFace lookup fails",
    )

    parser.add_argument(
        "--max-pool",
        type=int,
        default=DEFAULT_MAX_POOL,
        help=f"Maximum GPU combinations to consider (default: {DEFAULT_MAX_POOL})",
    )

    parser.add_argument(
        "--gpus-path",
        default=DEFAULT_GPUS_PATH,
        help=f"Path to GPUs JSON file (default: {DEFAULT_GPUS_PATH})",
    )

    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Only output export commands, no verbose analysis",
    )

    parser.add_argument(
        "--verbose",
        default=True,
        help="Show detailed GPU analysis and selection process",
    )

    parser.add_argument(
        "--restrict-gpus",
        help="Comma-separated list of GPU indices to restrict selection to (e.g., '0,2,3')",
    )

    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli()