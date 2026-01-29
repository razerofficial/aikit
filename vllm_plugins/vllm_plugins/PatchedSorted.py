import builtins
from vllm.v1.executor.ray_executor import RayWorkerMetaData


# =============================================================================
# 1 Custom key function factory
# =============================================================================
def sort_by_adjusted_rank(item):
    if isinstance(item, RayWorkerMetaData) and hasattr(item, "adjusted_rank"):
        return item.adjusted_rank
    else:
        return float("inf")

# =============================================================================
# 2 Class-based context manager to patch sorted()
# =============================================================================
class PatchedSorted:
    def __init__(self, target_gpu_order):
        self.target_gpu_order = target_gpu_order
        self._orig_sorted = builtins.sorted

    def __enter__(self):
        def patched_sorted(iterable, key=None, *args, **kwargs):
            print("Type of iterable:", type(iterable))
            print("iterable:", iterable)

            key_name = getattr(key, "__name__", None)

            #1 custom sort by ip_gpu
            if key_name == "sort_by_driver_then_worker_ip":
                import ray
                from typing import List

                worker_meta_list: List[RayWorkerMetaData] = list(iterable)
                node_gpu_ids = ray.get([
                    each.worker.get_node_and_gpu_ids.remote()
                    for each in worker_meta_list
                ])

                gpu_ids_only = [gpu_list for _, gpu_list in node_gpu_ids]

                for each, gpu_ids in zip(worker_meta_list, gpu_ids_only):
                    ip_gpu = each.ip + "_" + gpu_ids[0]

                    if ip_gpu in self.target_gpu_order:
                        each.adjusted_rank = self.target_gpu_order.index(ip_gpu)
                        print(f"sort_by_ip_gpu {ip_gpu} --> {each.adjusted_rank}")
                    else:
                        each.adjusted_rank = float("inf")

                worker_meta_list = self._orig_sorted(worker_meta_list, key=sort_by_adjusted_rank)
                return worker_meta_list

            #2 bypass GPU IDs sort
            if (key is None) and (all(isinstance(i, int) for i in iterable)):
                return list(iterable)

            #3 Fallback to normal behavior
            return self._orig_sorted(iterable, key=key, *args, **kwargs)

        builtins.sorted = patched_sorted
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Always restore original sorted
        builtins.sorted = self._orig_sorted

        # do not suppress exceptions
        return False
