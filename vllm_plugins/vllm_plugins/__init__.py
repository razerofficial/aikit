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

