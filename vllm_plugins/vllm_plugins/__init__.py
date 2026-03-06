def register_RayDistributedExecutor():
    try:
        from .patches import razer_ray_distributed_executor

        import os
        current_pid = os.getpid()
        print(f"[plugin] vLLM RayDistributedExecutor patch Successfully (PID: {current_pid})")

    except Exception as e:
        print("[plugin] vLLM RayDistributedExecutor patch failed:", e)

def register_MultiprocExecutor():
    try:
        from .patches import razer_multiproc_executor

        import os
        current_pid = os.getpid()
        print(f"[plugin] vLLM MultiprocExecutor patch Successfully (PID: {current_pid})")

    except Exception as e:
        print("[plugin] vLLM MultiprocExecutor patch failed:", e)

def register_RayWorkerWrapper():
    try:
        from vllm.v1.executor.ray_utils import RayWorkerWrapper

        import os
        current_pid = os.getpid()

        # already patched do nothing
        if hasattr(RayWorkerWrapper, "_org_initialize_from_config"):
            print(f"[plugin] vLLM RayWorkerWrapper patch already (PID: {current_pid})")
            return

        # not patched yet import module to apply patch
        from .patches import razer_ray_worker_wrapper

        print(f"[plugin] vLLM RayWorkerWrapper patch Successfully (PID: {current_pid})")

    except Exception as e:
        print("[plugin] vLLM RayWorkerWrapper patch failed:", e)

def register_GPUWorker():
    try:
        from .children import razer_gpu_worker

        import os
        current_pid = os.getpid()
        print(f"[plugin] vLLM GPUWorker patch Successfully (PID: {current_pid})")

    except Exception as e:
        print("[plugin] vLLM GPUWorker patch failed:", e)

def register_CoreEngineProcManager():
    try:
        from .children import razer_CoreEngineProcManager

        import os
        current_pid = os.getpid()
        print(f"[plugin] vLLM CoreEngineProcManager patch Successfully (PID: {current_pid})")

    except Exception as e:
        print("[plugin] vLLM CoreEngineProcManager patch failed:", e)

