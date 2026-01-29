from typing import Dict, Any, List

# =============================================================================
# 1. Collect NVML GPU info (local function)
# =============================================================================
def collect_nvml_info() -> List[Dict[str, Any]]:
    """Collect static GPU info using NVML (including free memory snapshot)."""
    import pynvml

    pynvml.nvmlInit()
    num_gpus = pynvml.nvmlDeviceGetCount()
    gpu_nvml_info = []

    for i in range(num_gpus):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        uuid = str(pynvml.nvmlDeviceGetUUID(handle))
        bus_id = pynvml.nvmlDeviceGetPciInfo(handle).busId
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

        gpu_nvml_info.append({
            "device_id": i,
            "uuid": uuid,
            "bus_id": bus_id,
            "total_memory": mem_info.total,
            "free_memory": mem_info.free,
        })

    pynvml.nvmlShutdown()
    return gpu_nvml_info


# =============================================================================
# 2. Remote Ray task to collect GPU info on each node
# =============================================================================
import ray

@ray.remote
def get_gpu_info(verbose: bool = False) -> Dict[str, Any]:
    """Remote Ray task to collect GPU info on a specific node."""
    try:
        from ray.util import get_node_ip_address
        import traceback

        node_ip = get_node_ip_address()
        gpu_info = collect_nvml_info()

        if verbose:
            print(f"[{node_ip}] Found {len(gpu_info)} GPU(s)")

        return {node_ip: gpu_info}

    except Exception as e:
        import traceback
        error_info = {
            "error": f"Failed on node {get_node_ip_address()}: {e}",
            "traceback": traceback.format_exc(),
        }
        print(f"get_gpu_info : {error_info}")

        return {}

# =============================================================================
# 3. Collect GPU info from all nodes
# =============================================================================
def collect_cluster_gpu_info(verbose: bool = True) -> List[Dict[str, Any]]:
    """Run one get_gpu_info task per active Ray node using NodeAffinity."""
    import ray

    if not ray.is_initialized():
        ray.init(address="auto")

    try:
        nodes = [n for n in ray.nodes() if n["Alive"]]
        cluster_results: List[Dict[str, Any]] = []

        for node in nodes:
            future = get_gpu_info.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node["NodeID"],
                    soft=False
                )
            ).remote(verbose)

            gpu_result = ray.get(future)

            # Append only if result is non-empty and not an error
            if isinstance(gpu_result, dict) and gpu_result:
                cluster_results.append(gpu_result)
            elif verbose:
                print(f"[WARN] Skipping empty or invalid result from node {node['NodeManagerAddress']}")

        return cluster_results

    finally:
        # --- Always cleanly shut down Ray ---
        if verbose:
            print("collect_cluster_gpu_info ray.shutdown()")

        ray.shutdown()
