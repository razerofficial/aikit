import vllm.v1.engine.utils as utils_module
from vllm.v1.engine.utils import CoreEngineProcManager
from typing import List, Dict, Any, Tuple, Optional, Callable
from vllm.config import VllmConfig
from vllm.v1.executor.abstract import Executor


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


def razer_run_engine_core(*args,
                           dp_rank: int = 0,
                           local_dp_rank: int = 0,
                           **kwargs):
    import os
    pid = os.getpid()
    print(f"razer_run_engine_core pid = {pid}")

    env_val = os.getenv("VLLM_GPU_ORDER")
    if env_val:
        gpu_uuid_order = [x.strip() for x in env_val.split(",") if x.strip()]
        print(f"{gpu_uuid_order=}")

        from vllm_plugins.gpu_info import collect_nvml_info
        gpu_list = collect_nvml_info()
        print(f"gpu_list = {gpu_list}")

        uuid_map = reformat_gpu_info(gpu_list)
        print(f"uuid_map = {uuid_map}")

        target_gpu_order, gpu_count = build_target_gpu_order(uuid_map, gpu_uuid_order)
        print(f"target_gpu_order = {target_gpu_order} gpu_count = {gpu_count}")

        world_size = kwargs['vllm_config'].parallel_config.world_size
        print(f"world_size = {world_size}")

        if gpu_count >= world_size:
            os.environ["CUDA_VISIBLE_DEVICES"] = target_gpu_order
        else:
            print("CUDA_VISIBLE_DEVICES is NOT set")

    from vllm.v1.engine.core import EngineCoreProc
    EngineCoreProc.run_engine_core(*args, dp_rank=dp_rank, local_dp_rank=local_dp_rank, **kwargs)


class RazerCoreEngineProcManager(CoreEngineProcManager):
     def __init__(
        self,
        target_fn: Callable,
        local_engine_count: int,
        start_index: int,
        local_start_index: int,
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        client_handshake_address: Optional[str] = None,
    ):
        import os
        pid = os.getpid()
        print(f"RazerCoreEngineProcManager __init__ pid = {pid}")
        print(f"RazerCoreEngineProcManager target = {target_fn} pid = {pid}")
        print(f"RazerCoreEngineProcManager executor_class = {executor_class} pid = {pid}")

        from vllm.v1.executor.abstract import UniProcExecutor
        if executor_class is UniProcExecutor:
            print("Executor is exactly UniProcExecutor")
            super().__init__(razer_run_engine_core, local_engine_count, start_index, local_start_index, vllm_config, local_client, handshake_address, executor_class, log_stats, client_handshake_address)
        else:
            print("Executor is Not UniProcExecutor")
            super().__init__(target_fn, local_engine_count, start_index, local_start_index, vllm_config, local_client, handshake_address, executor_class, log_stats, client_handshake_address)


utils_module.CoreEngineProcManager = RazerCoreEngineProcManager
