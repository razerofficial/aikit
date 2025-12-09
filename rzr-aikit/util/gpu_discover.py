#!/usr/bin/env python3
"""
GPU Discovery Tool

Discovers GPU devices locally or across a Ray cluster, performs benchmarks,
and outputs comprehensive GPU information including memory, compute capability,
and tensor format support.
"""

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Optional

from accelerate.utils import convert_bytes

# Configuration constants
CUDA_DEVICE_ORDER = "PCI_BUS_ID"  # Stable PCI bus ordering
DEFAULT_BENCH_DIM = 1024
DEFAULT_BENCH_ITERS = 4
DEFAULT_BENCH_WARMUP = 3

# Set CUDA device ordering
os.environ["CUDA_DEVICE_ORDER"] = CUDA_DEVICE_ORDER

# Tensor format support mapping
TENSOR_FORMAT_MAP = {
    "FP64": 1.3,
    "FP32": 1.0,
    "FP16": 5.3,
    "BF16": 8.0,
    "FP8_E4M3": 8.9,
    "FP8_E5M2": 8.9,
    "FP6_E2M3": 10.0,
    "FP6_E3M2": 10.0,
    "FP4_E2M1": 10.0,
    "INT32": 1.0,
    "INT16": 1.0,
    "INT8": 6.1,
    "INT4": 8.0,
}


@dataclass
class GPUInfo:
    """GPU information dataclass with comprehensive device details."""
    
    idx: int
    name: str
    ip: str
    pci_bus_id: Optional[str]
    uuid: str
    compute_capability: Optional[float]
    memory: Optional[object]  # NVML memory info object
    gflops: Optional[float]
    tensor_core_formats: list[str]
    
    def to_dict(self) -> dict:
        """Convert GPUInfo to dictionary for JSON serialization."""
        return {
            "idx": self.idx,
            "name": self.name,
            "ip": self.ip,
            "pci_bus_id": self.pci_bus_id,
            "uuid": self.uuid,
            "compute_capability": self.compute_capability,
            "memory": self._format_memory(),
            "gflops": self.gflops,
            "tensor_core_formats": self.tensor_core_formats,
        }
    
    def _format_memory(self) -> Optional[dict]:
        """Format memory information for serialization."""
        if self.memory is None:
            return None
        
        if hasattr(self.memory, 'total') and hasattr(self.memory, 'free'):
            return {
                "total": self.memory.total,
                "free": self.memory.free,
                "used": self.memory.total - self.memory.free
            }
        
        return self.memory
    
    def get_memory_gb(self) -> tuple[Optional[float], Optional[float]]:
        """Get total and free memory in GB."""
        if not self.memory or not hasattr(self.memory, 'total'):
            return None, None
        
        total_gb = self.memory.total / (1024**3)
        free_gb = self.memory.free / (1024**3)
        return total_gb, free_gb


def gpus_to_json(gpu_list: list[GPUInfo]) -> str:
    """Convert a list of GPUInfo objects to JSON string."""
    return json.dumps([gpu.to_dict() for gpu in gpu_list], indent=2)


def get_supported_formats(compute_capability: Optional[float]) -> list[str]:
    """Get supported tensor formats based on compute capability."""
    if compute_capability is None:
        return []
    return [fmt for fmt, min_cc in TENSOR_FORMAT_MAP.items() 
            if compute_capability >= min_cc]


def normalize_pci_bus_id(bus_id: str) -> Optional[str]:
    """Normalize PCI bus ID to standard format (e.g., 00000001:02:00.0)."""
    if not bus_id:
        return None
    
    bus_id = bus_id.strip()
    
    # Already normalized
    if re.match(r"^[0-9a-f]{8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$", bus_id):
        return bus_id
    
    # Parse and normalize
    match = re.match(r"^([0-9A-Fa-f]+):([0-9A-Fa-f]{2}):([0-9A-Fa-f]{2})\.([0-7])$", bus_id)
    if not match:
        return bus_id.lower()
    
    domain, bus, device, func = match.groups()
    return f"{domain.zfill(8).lower()}:{bus.lower()}:{device.lower()}.{func}"


class GPUBenchmark:
    """Handles GPU benchmarking operations."""
    
    def __init__(self, dim: int = DEFAULT_BENCH_DIM, 
                 iters: int = DEFAULT_BENCH_ITERS, 
                 warmup: int = DEFAULT_BENCH_WARMUP):
        self.dim = dim
        self.iters = iters
        self.warmup = warmup
    
    def run_fp16_benchmark(self, device_index: int) -> Optional[float]:
        """Run FP16 GEMM benchmark on specified device."""
        # Try PyTorch first
        gflops = self._try_pytorch_benchmark(device_index)
        if gflops is not None:
            return gflops
        
        # Fallback to CuPy
        return self._try_cupy_benchmark(device_index)
    
    def _try_pytorch_benchmark(self, device_index: int) -> Optional[float]:
        """Attempt PyTorch-based benchmark."""
        try:
            import torch
            
            if not torch.cuda.is_available():
                return None
            
            device = torch.device(f"cuda:{device_index}")
            dtype = torch.float16
            
            # Create matrices
            a = torch.randn((self.dim, self.dim), device=device, dtype=dtype)
            b = torch.randn((self.dim, self.dim), device=device, dtype=dtype)
            
            # Warmup
            for _ in range(self.warmup):
                a @ b
            torch.cuda.synchronize(device)
            
            # Benchmark
            start_time = time.perf_counter()
            for _ in range(self.iters):
                a @ b
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start_time
            
            # Calculate GFLOPS
            flops = 2.0 * (self.dim ** 3) * self.iters
            gflops = flops / elapsed / 1e9
            
            # Cleanup
            del a, b
            torch.cuda.empty_cache()
            return gflops
            
        except Exception:
            return None
    
    def _try_cupy_benchmark(self, device_index: int) -> Optional[float]:
        """Attempt CuPy-based benchmark."""
        try:
            import cupy as cp
            
            with cp.cuda.Device(device_index):
                # Create matrices
                a = cp.random.standard_normal((self.dim, self.dim), dtype=cp.float16)
                b = cp.random.standard_normal((self.dim, self.dim), dtype=cp.float16)
                
                # Warmup
                for _ in range(self.warmup):
                    a.dot(b)
                cp.cuda.runtime.deviceSynchronize()
                
                # Benchmark with events
                start_event = cp.cuda.Event()
                end_event = cp.cuda.Event()
                
                start_event.record()
                for _ in range(self.iters):
                    a.dot(b)
                end_event.record()
                end_event.synchronize()
                
                elapsed = cp.cuda.get_elapsed_time(start_event, end_event) / 1000.0
                
                # Calculate GFLOPS
                flops = 2.0 * (self.dim ** 3) * self.iters
                gflops = flops / elapsed / 1e9
                
                # Cleanup
                del a, b
                cp.get_default_memory_pool().free_all_blocks()
                return gflops
                
        except Exception:
            return None


class PlacementGroupManager:
    """Manages Ray placement groups for GPU filtering."""
    
    @staticmethod
    def _get_pg_name(ip: str, local_idx: int) -> str:
        """Generate placement group name matching gpu-filter format."""
        return f"park_{ip.replace('.', '_')}_gpu_{local_idx}"
    
    @staticmethod
    def cleanup_filter_placement_groups() -> None:
        """Remove placement groups created by gpu-filter."""
        try:
            import ray
            from ray.util.placement_group import remove_placement_group
        except ImportError:
            return
        
        try:
            gpu_nodes = [n for n in ray.nodes() 
                        if n.get("Alive") and n.get("Resources", {}).get("GPU", 0) > 0]
            
            for node in gpu_nodes:
                ip = node["NodeManagerAddress"]
                gpu_count = int(node["Resources"].get("GPU", 0))
                
                for local_idx in range(gpu_count):
                    pg_name = PlacementGroupManager._get_pg_name(ip, local_idx)
                    
                    try:
                        old_pg = ray.util.get_placement_group(pg_name)
                        if old_pg:
                            remove_placement_group(old_pg)
                    except Exception:
                        continue
                        
        except Exception:
            pass


class GPUDiscovery:
    """Handles local and distributed GPU discovery operations."""
    
    def __init__(self, benchmark_config: Optional[dict] = None):
        self.benchmark_config = benchmark_config or {}
        self.benchmarker = GPUBenchmark(
            dim=self.benchmark_config.get('dim', DEFAULT_BENCH_DIM),
            iters=self.benchmark_config.get('iters', DEFAULT_BENCH_ITERS),
            warmup=self.benchmark_config.get('warmup', DEFAULT_BENCH_WARMUP)
        )
    
    @staticmethod
    def _make_all_gpus_visible() -> Optional[str]:
        """Make all GPUs visible and return original CUDA_VISIBLE_DEVICES value."""
        try:
            import pynvml as nvml
            nvml.nvmlInit()
            device_count = nvml.nvmlDeviceGetCount()
            nvml.nvmlShutdown()
            
            original_cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
            if device_count > 0:
                all_devices = ",".join(str(i) for i in range(device_count))
                os.environ["CUDA_VISIBLE_DEVICES"] = all_devices
            
            return original_cuda_devices
        except Exception:
            return os.environ.get("CUDA_VISIBLE_DEVICES")
    
    @staticmethod
    def _restore_cuda_visibility(original_value: Optional[str]):
        """Restore original CUDA_VISIBLE_DEVICES value."""
        if original_value is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = original_value
        elif "CUDA_VISIBLE_DEVICES" in os.environ:
            del os.environ["CUDA_VISIBLE_DEVICES"]
    
    def discover_local(self, ip_addr: str = "", name_suffix: str = "", 
                      enable_bench: bool = True) -> list[GPUInfo]:
        """Discover GPUs on the local machine using NVML."""
        try:
            import pynvml as nvml
        except ImportError:
            raise ImportError("pynvml is required for GPU discovery")
        
        # Save original CUDA_VISIBLE_DEVICES and temporarily make all GPUs visible for discovery
        original_cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        
        try:
            nvml.nvmlInit()
            device_count = nvml.nvmlDeviceGetCount()
            
            # Make all GPUs visible for benchmarking
            if enable_bench and device_count > 0:
                all_devices = ",".join(str(i) for i in range(device_count))
                os.environ["CUDA_VISIBLE_DEVICES"] = all_devices
            
            gpus = []
            for i in range(device_count):
                gpu_info = self._get_device_info(nvml, i, ip_addr, name_suffix, enable_bench)
                if gpu_info:
                    gpus.append(gpu_info)
            
            nvml.nvmlShutdown()
            # Sort by PCI bus ID for consistent ordering
            gpus.sort(key=lambda x: x.pci_bus_id or "")
            return gpus
            
        except Exception as e:
            print(f"Error discovering local GPUs: {e}")
            return []
        finally:
            # Restore original CUDA_VISIBLE_DEVICES
            if original_cuda_devices is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_devices
            elif "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]
    
    def _get_device_info(self, nvml, device_index: int, ip_addr: str, 
                        name_suffix: str, enable_bench: bool) -> Optional[GPUInfo]:
        """Get comprehensive information for a single GPU device."""
        try:
            handle = nvml.nvmlDeviceGetHandleByIndex(device_index)
            
            # Basic device info
            name = self._decode_bytes(nvml.nvmlDeviceGetName(handle))
            if name_suffix:
                name = f"{name} {name_suffix}"
            
            # PCI info
            pci_info = nvml.nvmlDeviceGetPciInfo(handle)
            bus_id = self._decode_bytes(pci_info.busId)
            normalized_bus = normalize_pci_bus_id(bus_id)
            
            # UUID and compute capability
            uuid = self._decode_bytes(nvml.nvmlDeviceGetUUID(handle))
            maj, min_ver = nvml.nvmlDeviceGetCudaComputeCapability(handle)
            compute_cap = float(f"{maj}.{min_ver}")
            
            # Memory info and formats
            memory_info = nvml.nvmlDeviceGetMemoryInfo(handle)
            tensor_formats = get_supported_formats(compute_cap)
            
            # Benchmark if enabled and supported
            gflops = None
            if enable_bench and compute_cap >= 5.3:  # FP16 support threshold
                # Use device_index directly since we made all GPUs visible
                gflops = self.benchmarker.run_fp16_benchmark(device_index)
            
            return GPUInfo(
                idx=device_index,
                name=name,
                ip=ip_addr,
                pci_bus_id=normalized_bus,
                uuid=uuid,
                compute_capability=compute_cap,
                memory=memory_info,
                gflops=gflops,
                tensor_core_formats=tensor_formats,
            )
            
        except Exception as e:
            print(f"Error getting info for device {device_index}: {e}")
            return None
    
    @staticmethod
    def _decode_bytes(value) -> str:
        """Decode bytes to string if needed."""
        return value.decode() if isinstance(value, bytes) else str(value)


    def discover_distributed(self, enable_bench: bool = True) -> list[GPUInfo]:
        """Discover GPUs across a Ray cluster."""
        try:
            import ray
            from ray.util import get_node_ip_address
        except ImportError:
            raise ImportError("Ray is required for distributed discovery")
        
        # Initialize Ray connection
        try:
            ray.init(address="auto", ignore_reinit_error=True, namespace="parking")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Ray cluster: {e}")
        
        # Clean up existing placement groups
        PlacementGroupManager.cleanup_filter_placement_groups()
        
        # Get GPU nodes
        gpu_nodes = [n for n in ray.nodes() 
                    if n.get("Alive") and n.get("Resources", {}).get("GPU", 0) > 0]
        
        if not gpu_nodes:
            print("No Ray nodes with GPUs detected; falling back to local discovery")
            return self.discover_local()
        
        local_ip = get_node_ip_address()
        
        # Remote functions for distributed discovery
        @ray.remote(num_cpus=0)
        def inventory_node(ip: str, local_ip_ref: str):
            suffix = f"({ip}{' [local]' if ip == local_ip_ref else ''})"
            discovery = GPUDiscovery(self.benchmark_config)
            return {
                "ip": ip,
                "items": discovery.discover_local(ip_addr=ip, name_suffix=f" {suffix}", 
                                                enable_bench=False)
            }
        
        @ray.remote(num_cpus=0, num_gpus=1)
        def benchmark_gpu():
            return self._get_distributed_benchmark_info()
        
        # Collect inventory from all nodes
        inventories = self._collect_node_inventories(gpu_nodes, local_ip, inventory_node)
        
        # Collect benchmarks if enabled
        benchmark_results = {}
        if enable_bench:
            benchmark_results = self._collect_benchmark_results(
                gpu_nodes, inventories, benchmark_gpu)
        
        # Combine results with local node first
        return self._combine_distributed_results(
            inventories, benchmark_results, local_ip, gpu_nodes)
    
    def _collect_node_inventories(self, gpu_nodes: list, local_ip: str, inventory_func) -> list:
        """Collect GPU inventories from all nodes."""
        import ray
        
        inventory_tasks = []
        for node in gpu_nodes:
            ip = node["NodeManagerAddress"]
            node_resources = self._get_node_resources(node, ip)
            task = inventory_func.options(**node_resources).remote(ip, local_ip)
            inventory_tasks.append(task)
        
        return ray.get(inventory_tasks) if inventory_tasks else []
    
    def _collect_benchmark_results(self, gpu_nodes: list, inventories: list, 
                                 benchmark_func) -> dict:
        """Collect benchmark results from distributed GPUs."""
        import ray
        
        benchmark_tasks = []
        for inventory in inventories:
            ip = inventory["ip"]
            gpu_count = len(inventory["items"])
            node = next((n for n in gpu_nodes if n["NodeManagerAddress"] == ip), None)
            
            if node:
                node_resources = self._get_node_resources(node, ip)
                for _ in range(gpu_count):
                    task = benchmark_func.options(**node_resources).remote()
                    benchmark_tasks.append(task)
        
        results = {}
        if benchmark_tasks:
            for result in ray.get(benchmark_tasks):
                if result and result.get("ip") and result.get("bus_id"):
                    key = (result["ip"], result["bus_id"])
                    results[key] = result.get("gflops")
        
        return results
    
    def _get_distributed_benchmark_info(self) -> dict:
        """Get benchmark info for current GPU in distributed setting."""
        import os
        import subprocess
        from ray.util import get_node_ip_address
        
        # Save original CUDA_VISIBLE_DEVICES and temporarily make all GPUs visible
        original_cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        
        try:
            # For distributed benchmark, we need to ensure the assigned GPU is accessible
            # Ray should have already set CUDA_VISIBLE_DEVICES to the assigned GPU
            # But we need to discover which physical GPU this corresponds to
            
            node_ip = get_node_ip_address()
            bus_id = self._get_current_gpu_bus_id()
            normalized_bus = normalize_pci_bus_id(bus_id) if bus_id else None
            
            # Run benchmark on device 0 (which should be the only visible GPU in distributed mode)
            gflops = self.benchmarker.run_fp16_benchmark(0)
            
            return {
                "ip": node_ip,
                "bus_id": normalized_bus,
                "gflops": gflops
            }
        finally:
            # Restore original CUDA_VISIBLE_DEVICES if it was modified
            if original_cuda_devices is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_devices
            elif "CUDA_VISIBLE_DEVICES" in os.environ and os.environ["CUDA_VISIBLE_DEVICES"] != original_cuda_devices:
                if original_cuda_devices is None:
                    del os.environ["CUDA_VISIBLE_DEVICES"]
    
    def _get_current_gpu_bus_id(self) -> Optional[str]:
        """Get PCI bus ID of current CUDA device 0."""
        # Try CuPy first
        try:
            import cupy as cp
            bus_id = cp.cuda.runtime.deviceGetPCIBusId(0)
            return bus_id.decode() if isinstance(bus_id, (bytes, bytearray)) else str(bus_id)
        except Exception:
            pass
        
        # Fallback to nvidia-smi
        try:
            import os
            import subprocess
            
            visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            if visible_devices and visible_devices[0].strip():
                result = subprocess.check_output([
                    "nvidia-smi", "-i", visible_devices[0].strip(),
                    "--query-gpu=pci.bus_id", "--format=csv,noheader"
                ], text=True).strip()
                
                if result:
                    return result.splitlines()[0].strip()
        except Exception:
            pass
        
        return None
    
    @staticmethod
    def _get_node_resources(node: dict, ip: str) -> dict:
        """Get resource constraints for targeting specific node."""
        for resource_key in node["Resources"].keys():
            if resource_key.startswith("node:") and resource_key.split("node:", 1)[1] == ip:
                return {"resources": {resource_key: 0.001}}
        return {}
    
    def _combine_distributed_results(
        self, 
        inventories: list, 
        benchmark_results: dict,
        local_ip: str, 
        gpu_nodes: list
    ) -> list[GPUInfo]:
        """Combine GPU inventories and benchmarks with vLLM-compatible ordering.
        
        Orders GPUs to match vLLM's worker placement strategy for distributed inference.
        This ensures GPU indices in discovery match vLLM's tensor parallel rank assignments.
        
        Ordering logic (matches vLLM's sort_by_driver_then_worker_ip):
            1. Driver/head node GPUs first
            2. Worker nodes with fewer GPUs (for balanced distribution)
            3. Lexicographic IP ordering (for deterministic results)
        
        Args:
            inventories: List of GPU inventories per node
            benchmark_results: Performance benchmark results keyed by (ip, pci_bus_id)
            local_ip: IP address of the head/driver node
            gpu_nodes: Ray cluster node information (unused, kept for compatibility)
        
        Returns:
            Ordered list of GPUInfo objects with updated indices and benchmark results
        """
        # Group GPUs by node IP address
        gpus_by_ip = {inv["ip"]: inv["items"] for inv in inventories}
        
        # Count GPUs per node for sorting
        gpu_count_by_ip = {ip: len(gpus) for ip, gpus in gpus_by_ip.items()}
        
        # Sort node IPs using vLLM's ordering strategy
        def vllm_sort_key(ip: str) -> tuple[int, int, str]:
            """Generate sort key matching vLLM's sort_by_driver_then_worker_ip logic."""
            is_driver = 0 if ip == local_ip else 1
            gpu_count = gpu_count_by_ip[ip]
            return (is_driver, gpu_count, ip)
        
        ordered_ips = sorted(gpus_by_ip.keys(), key=vllm_sort_key)
        
        # Flatten GPUs in sorted node order
        combined_gpus = []
        for ip in ordered_ips:
            for gpu in gpus_by_ip[ip]:
                # Attach benchmark results if available
                benchmark_key = (ip, gpu.pci_bus_id)
                if benchmark_key in benchmark_results:
                    gpu.gflops = benchmark_results[benchmark_key]
                
                combined_gpus.append(gpu)
        
        # Reassign global GPU indices
        for idx, gpu in enumerate(combined_gpus):
            gpu.idx = idx
        
        return combined_gpus


def main(args: argparse.Namespace):
    """Main discovery function."""
    # Initialize discovery with benchmark configuration
    benchmark_config = {
        'dim': args.bench_dim,
        'iters': args.bench_iters,
        'warmup': args.bench_warmup
    }
    
    discovery = GPUDiscovery(benchmark_config)
    
    # Discover GPUs
    if args.distributed:
        gpus = discovery.discover_distributed(enable_bench=args.bench)
    else:
        gpus = discovery.discover_local(enable_bench=args.bench)
        # Reindex for local discovery
        for idx, gpu in enumerate(gpus):
            gpu.idx = idx
    
    # Save to JSON file
    with open("gpus.json", "w") as f:
        f.write(gpus_to_json(gpus))
    
    # Display results
    _display_gpu_info(gpus)


def _display_gpu_info(gpus: list[GPUInfo]):
    """Display formatted GPU information."""
    for gpu in gpus:
        title = f"[{gpu.idx}] {gpu.name}"
        print(title)
        print("-" * len(title))
        
        print(f"  IP Address: {gpu.ip or 'N/A'}")
        print(f"  PCI Bus ID: {gpu.pci_bus_id or 'N/A'}")
        print(f"  UUID: {gpu.uuid}")
        print(f"  Compute Capability: {gpu.compute_capability or 'N/A'}")
        
        # Format memory information
        memory_info = _format_memory_display(gpu.memory)
        print(f"  Memory (Total | Free): {memory_info}")
        
        # Format GFLOPS
        gflops_info = "N/A" if gpu.gflops is None else f"{gpu.gflops:,.2f}"
        print(f"  GFLOPS (FP16): {gflops_info}")
        
        # Format tensor formats
        formats = ", ".join(gpu.tensor_core_formats) if gpu.tensor_core_formats else "none"
        print(f"  Tensor Core Formats: {formats}")
        print()


def _format_memory_display(memory_info) -> str:
    """Format memory information for display."""
    if memory_info is None:
        return "N/A | N/A"
    
    if hasattr(memory_info, 'total') and hasattr(memory_info, 'free'):
        total_str = convert_bytes(memory_info.total)
        free_str = convert_bytes(memory_info.free)
        return f"{total_str} | {free_str}"
    
    return "N/A | N/A"


def cli():
    """Command-line interface for GPU discovery."""
    parser = argparse.ArgumentParser(
        prog="gpu-discovery",
        description="Discover local or distributed GPUs with comprehensive device information.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Discover GPUs across a Ray cluster"
    )

    # Benchmark options
    bench_group = parser.add_mutually_exclusive_group()
    bench_group.add_argument(
        "--bench",
        action="store_true",
        default=True,
        help="Enable FP16 benchmark"
    )
    bench_group.add_argument(
        "--no-bench",
        action="store_false",
        dest="bench",
        help="Disable FP16 benchmark"
    )

    parser.add_argument(
        "--bench-dim",
        type=int,
        default=DEFAULT_BENCH_DIM,
        help="Matrix dimension for GEMM benchmark"
    )
    parser.add_argument(
        "--bench-iters",
        type=int,
        default=DEFAULT_BENCH_ITERS,
        help="Number of benchmark iterations"
    )
    parser.add_argument(
        "--bench-warmup",
        type=int,
        default=DEFAULT_BENCH_WARMUP,
        help="Number of warmup iterations"
    )

    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli()
