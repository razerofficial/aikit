from vllm.v1.executor.ray_distributed_executor import RayDistributedExecutor
from typing import Any, Callable, Optional, Union
from concurrent.futures import Future


RayDistributedExecutor._org_init_ = RayDistributedExecutor.__init__
RayDistributedExecutor._org_init_executor = RayDistributedExecutor._init_executor
RayDistributedExecutor._org_init_workers_ray = RayDistributedExecutor._init_workers_ray
RayDistributedExecutor._org_collective_rpc = RayDistributedExecutor.collective_rpc


def reformat_gpu_info(cluster_gpus: list[dict[str, any]]):
    """Flatten and reindex cluster GPU info by UUID."""
    uuid_map = {}

    for node_entry in cluster_gpus:
        for ip, gpu_list in node_entry.items():
            for gpu in gpu_list:
                uuid = gpu["uuid"]
                uuid_map[uuid] = {
                    "ip_address": ip,
                    "device_id": gpu["device_id"],
                    "bus_id": gpu["bus_id"],
                    "total_memory": gpu["total_memory"],
                    "free_memory": gpu["free_memory"],
                }

    return uuid_map


def build_target_gpu_order(uuid_map: dict, gpu_uuid_order: list):
    """Convert UUID order to target GPU order like ip_deviceID."""
    target_gpu_order = []

    for uuid in gpu_uuid_order:
        if uuid in uuid_map:
            ip = uuid_map[uuid]["ip_address"]
            dev = uuid_map[uuid]["device_id"]
            target_gpu_order.append(f"{ip}_{dev}")
        else:
            print(f"[WARN] UUID {uuid} not found in cluster info")

    return target_gpu_order


def __init__(self, *args, **kwargs):
    self.target_gpu_order = None

    import os
    os.environ.setdefault("RAY_CGRAPH_get_timeout", "300")
    env_val = os.getenv("VLLM_GPU_ORDER")
    if env_val:
        from .. import register_RayWorkerWrapper
        register_RayWorkerWrapper()

        gpu_uuid_order = [x.strip() for x in env_val.split(",") if x.strip()]
        print(f"{gpu_uuid_order=}")

        # convert UUID to IP_GPU
        from vllm_plugins.gpu_info import collect_cluster_gpu_info
        gpu_list = collect_cluster_gpu_info(verbose=True)
        print(f"gpu_list = {gpu_list}")

        uuid_map = reformat_gpu_info(gpu_list)
        print(f"uuid_map = {uuid_map}")

        temp_target_gpu_order = build_target_gpu_order(uuid_map, gpu_uuid_order)
        print(f"temp_target_gpu_order = {temp_target_gpu_order}")

        if len(temp_target_gpu_order) > 0:
            from vllm.config import VllmConfig
            if args and isinstance(args[0], VllmConfig):
                vllm_config = args[0]
                world_size = vllm_config.parallel_config.world_size

                if len(temp_target_gpu_order) >= world_size:
                    self.target_gpu_order = temp_target_gpu_order
                    print(f"razer_ray_distributed_executor __init__ target_gpu_order : {self.target_gpu_order}")
                else:
                    print("razer_ray_distributed_executor __init__ target_gpu_order is NOT set")

        print(f"self.target_gpu_order = {self.target_gpu_order}")

    print(f"razer_ray_distributed_executor __init__ _org_init_ = {self._org_init_}")
    self._org_init_(*args, **kwargs)


def _init_executor(self) -> None:
    if self.target_gpu_order:
        from ray.runtime_env import RuntimeEnv

        base_env = self.parallel_config.ray_runtime_env
        if base_env is None:
            runtime_env = RuntimeEnv()
            runtime_env["worker_process_setup_hook"] = "vllm_plugins.register_RayWorkerWrapper"
            self.parallel_config.ray_runtime_env = runtime_env
        else:
            base_env["worker_process_setup_hook"] = "vllm_plugins.register_RayWorkerWrapper"

    print(f"razer_ray_distributed_executor _init_executor _org_init_executor = {self._org_init_executor}")
    self._org_init_executor()


def _init_workers_ray(self, placement_group: "PlacementGroup", **ray_remote_kwargs):
    print(f"razer_ray_distributed_executor _init_workers_ray _org_init_workers_ray = {self._org_init_workers_ray}")

    if self.target_gpu_order:
        print(f"razer_ray_distributed_executor _init_workers_ray target_gpu_order : {self.target_gpu_order}")
        from vllm_plugins.PatchedSorted import PatchedSorted
        with PatchedSorted(self.target_gpu_order):
            self._org_init_workers_ray(placement_group, **ray_remote_kwargs)
    else:
        self._org_init_workers_ray(placement_group, **ray_remote_kwargs)


def collective_rpc(
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
        non_block: bool = False,
    ) -> list[Any] | Future[list[Any]]:
    print(f"razer_ray_distributed_executor collective_rpc _org_collective_rpc = {self._org_collective_rpc}")

    if isinstance(method, str):
        print(f"collective_rpc method is string : {method}")
        if method == "update_environment_variables":
            result = self._org_collective_rpc(method,
                                              timeout=timeout,
                                              args=args,
                                              kwargs=kwargs,
                                              non_block=non_block)
            return result

    return self._org_collective_rpc(method,
                                    timeout=timeout,
                                    args=args,
                                    kwargs=kwargs,
                                    non_block=non_block)


RayDistributedExecutor.__init__ = __init__
RayDistributedExecutor._init_executor = _init_executor
RayDistributedExecutor._init_workers_ray = _init_workers_ray
RayDistributedExecutor.collective_rpc = collective_rpc
