from typing import List, Dict, Any, Tuple
from vllm.config import VllmConfig
from vllm.v1.executor.multiproc_executor import MultiprocExecutor


MultiprocExecutor._org_init_ = MultiprocExecutor.__init__


def reformat_gpu_info(gpu_list: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Convert a list of GPU info dicts into a dict keyed by UUID.
    """
    uuid_map = {}
    for gpu in gpu_list:
        uuid_map[gpu["uuid"]] = {
            "device_id": gpu["device_id"],
            "bus_id": gpu["bus_id"],
            "total_memory": gpu.get("total_memory"),
            "free_memory": gpu.get("free_memory"),
        }
    return uuid_map


def build_target_gpu_order(uuid_map: Dict[str, Dict[str, Any]], gpu_uuid_order: List[str]) -> Tuple[str, int]:
    """
    Convert UUID order list into a CUDA_VISIBLE_DEVICES-compatible string.
    Example output: "1,0,2"
    """
    device_ids: List[int] = []

    for uuid in gpu_uuid_order:
        if uuid in uuid_map:
            dev_id = uuid_map[uuid]["device_id"]
            device_ids.append(dev_id)
        else:
            print(f"[WARN] UUID {uuid} not found in cluster info")

    # Convert list of ints to comma-separated string
    device_str = ",".join(str(i) for i in device_ids)
    return device_str, len(device_ids)


def __init__(
    self,
    vllm_config: VllmConfig,
) -> None:
    import os
    env_val = os.getenv("VLLM_GPU_ORDER")
    if env_val:
        gpu_uuid_order = [x.strip() for x in env_val.split(",") if x.strip()]

        from vllm_plugins.gpu_info import collect_nvml_info
        gpu_list = collect_nvml_info()
        print(gpu_list)

        uuid_map = reformat_gpu_info(gpu_list)
        print(uuid_map)

        target_gpu_order, gpu_count = build_target_gpu_order(uuid_map, gpu_uuid_order)
        print(f"target_gpu_order : {target_gpu_order} gpu_count : {gpu_count}")

        world_size = vllm_config.parallel_config.world_size
        print(f"world_size = {world_size}")

        if gpu_count >= world_size:
            os.environ["CUDA_VISIBLE_DEVICES"] = target_gpu_order
        else:
            print("CUDA_VISIBLE_DEVICES is NOT set")

    self._org_init_(vllm_config)


MultiprocExecutor.__init__ = __init__
