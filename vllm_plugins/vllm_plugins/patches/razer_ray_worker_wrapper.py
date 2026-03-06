from vllm.v1.executor.ray_utils import RayWorkerWrapper
from typing import Any


RayWorkerWrapper._org_initialize_from_config = RayWorkerWrapper.initialize_from_config


def razer_initialize_from_config(self, kv_cache_configs: list[Any]) -> None:
    # replace global_rank with rpc_rank
    org_global_rank = self.global_rank
    self.global_rank = self.rpc_rank

    # call original function
    self._org_initialize_from_config(kv_cache_configs)

    # restore global_rank
    self.global_rank = org_global_rank


RayWorkerWrapper.initialize_from_config = razer_initialize_from_config
print(f"patched RazerRayWorkerWrapper")
