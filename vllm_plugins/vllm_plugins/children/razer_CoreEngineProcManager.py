import inspect
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


def razer_run_engine_core(vllm_config: VllmConfig):
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

        world_size = vllm_config.parallel_config.world_size
        print(f"world_size = {world_size}")

        if gpu_count >= world_size:
            os.environ["CUDA_VISIBLE_DEVICES"] = target_gpu_order
        else:
            print("CUDA_VISIBLE_DEVICES is NOT set")


class RazerCoreEngineProcManager(CoreEngineProcManager):
     def __init__(self, *args, **kwargs):
        import os
        pid = os.getpid()
        print(f"RazerCoreEngineProcManager __init__ pid = {pid}")

        # Dynamically bind the passed arguments to the base class signature
        sig = inspect.signature(CoreEngineProcManager.__init__)

        # self must be passed because __init__ expects it
        bound_args = sig.bind(self, *args, **kwargs)

        # Apply defaults in case they weren't explicitly passed
        bound_args.apply_defaults()

        # Safely extract exactly the arguments required by name
        vllm_config = bound_args.arguments.get('vllm_config')
        executor_class = bound_args.arguments.get('executor_class')

        print(f"RazerCoreEngineProcManager executor_class = {executor_class} pid = {pid}")

        from vllm.v1.executor.abstract import UniProcExecutor
        if executor_class is UniProcExecutor:
            print("Executor is exactly UniProcExecutor")

            if vllm_config:
                razer_run_engine_core(vllm_config)

            super().__init__(*args, **kwargs)
        else:
            print("Executor is Not UniProcExecutor")
            super().__init__(*args, **kwargs)


utils_module.CoreEngineProcManager = RazerCoreEngineProcManager
