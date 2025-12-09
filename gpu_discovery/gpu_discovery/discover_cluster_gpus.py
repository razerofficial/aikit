from typing import List, Dict, Any
import ray

@ray.remote
def save_node_cache(gpus: List[Dict[str, Any]]):
    """Save GPUs info on the node where this task runs."""
    from .cache import save_cache
    save_cache(gpus)
    return True

@ray.remote(num_gpus=1)
def get_single_gpu_info(verbose: bool = False) -> Dict[str, Any]:
    import os
    import torch
    from ray.util import get_node_ip_address
    from .discover_local_gpus import discover_local_gpus

    ctx = ray.get_runtime_context()
    node_id = ctx.get_node_id()
    node_ip = get_node_ip_address()

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)

    physical_id = -1
    if cuda_visible:
        physical_id = int(cuda_visible)

    if verbose:
        # Debug: show environment variables relevant to GPU assignment
        print(f"[DEBUG] Node {node_ip} CUDA_VISIBLE_DEVICES={cuda_visible}")

        # Debug: torch GPU visibility
        num_gpus = torch.cuda.device_count()
        print(f"[DEBUG] Node {node_ip} torch.cuda.device_count()={num_gpus}")
        for i in range(num_gpus):
            print(f"[DEBUG] Node {node_ip} GPU {i}: {torch.cuda.get_device_name(i)}")

    gpus = discover_local_gpus(verbose)

    if verbose:
        import json
        print(f"Ray GPUs before set device id at Node {node_ip}")
        print(json.dumps(gpus, indent=2))

    # Set device id
    if gpus and physical_id != -1:
        gpus[0]['device_id'] = physical_id

    if verbose:
        import json
        print(f"Ray GPUs after set device id at Node {node_ip}")
        print(json.dumps(gpus, indent=2))

    return {
        "node_id": node_id,
        "node_ip": node_ip,
        "gpus": gpus
    }

def discover_cluster_gpus(verbose: bool = False) -> List[Dict[str, Any]]:
    """
    Discover GPUs across all nodes in the Ray cluster.
    Runs one Ray task per GPU (safe for CUDA kernel benchmarks).
    Aggregates per-node results into a clean list of dicts.
    """
    # Get all alive nodes
    nodes = [n for n in ray.nodes() if n["Alive"]]

    cluster_results: List[Dict[str, Any]] = []

    for node in nodes:
        n_gpus = int(node.get("Resources", {}).get("GPU", 0))
        if n_gpus <= 0:
            continue

        # Launch one task per GPU
        futures = [
            get_single_gpu_info.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node["NodeID"],
                    soft=False
                )
            ).remote(verbose)
            for _ in range(n_gpus)
        ]

        gpu_results = ray.get(futures)

        # Collect all GPU entries from this node
        gpus_per_node = []
        for r in gpu_results:
            gpus_per_node.extend(r["gpus"])

        # Sort the list of GPU dictionaries based on the 'device_id' key in ascending order.
        sorted_gpus_per_node = sorted(gpus_per_node, key=lambda gpu: gpu['device_id'])

        cluster_results.append({
            "node_id": node["NodeID"],
            "node_ip": node["NodeManagerAddress"],
            "gpus": sorted_gpus_per_node
        })

        # Save GPUs info for this node
        if sorted_gpus_per_node:
            ray.get(
                save_node_cache.options(
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node["NodeID"],
                        soft=False
                    )
                ).remote(sorted_gpus_per_node)
            )

            if verbose:
                import json
                print(f"Save GPUs info for node : {node["NodeManagerAddress"]}")
                print(json.dumps(sorted_gpus_per_node, indent=2))

    return cluster_results
