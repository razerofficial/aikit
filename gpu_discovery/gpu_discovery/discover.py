import os
import fcntl

from .discover_local_gpus import discover_local_gpus, refresh_nvml_info


class SimpleCrashSafeLock:
    """
    Crash-safe file lock with class-level lock file path.
    Lock is automatically released by OS when program crashes.
    """

    lock_file = "/tmp/discover_local_gpus.lock"

    def __init__(self, verbose: bool = False):
        self._lock_fd = None
        self._verbose = verbose

    @classmethod
    def set_lock_file(cls, path: str):
        """Set custom lock file path for all instances."""
        cls.lock_file = path

    def acquire(self) -> bool:
        """Acquire lock non-blocking. Returns True if successful."""
        if self._lock_fd is not None:
            # Already locked by this instance
            return True

        try:
            self._lock_fd = os.open(self.lock_file, os.O_CREAT | os.O_WRONLY, 0o644)
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

            if self._verbose:
                pid = os.getpid()
                print(f"SimpleCrashSafeLock is acquired by process {pid} successfully")

            return True
        except (IOError, OSError, BlockingIOError):
            if self._verbose:
                pid = os.getpid()
                print(f"Fail to acquire SimpleCrashSafeLock by process {pid}")

            if self._lock_fd is not None:
                os.close(self._lock_fd)
                self._lock_fd = None
            return False

    def release(self) -> bool:
        """Release lock and delete lock file."""
        if self._lock_fd is None:
            return False

        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            os.close(self._lock_fd)
            os.unlink(self.lock_file)
            self._lock_fd = None

            if self._verbose:
                pid = os.getpid()
                print(f"SimpleCrashSafeLock is released by process {pid} successfully")

            return True
        except (IOError, OSError):
            if self._verbose:
                pid = os.getpid()
                print(f"Fail to release SimpleCrashSafeLock by process {pid}")

            return False

    def __enter__(self):
        if not self.acquire():
            raise RuntimeError("Could not acquire lock - another instance may be running")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False


# ---------- Ray Actor Lock for Cluster ----------
try:
    import ray

    # -------------------------
    # Cluster-wide lock (Ray)
    # -------------------------
    @ray.remote
    class DiscoveryLock:
        """Cluster-wide mutex for GPU discovery."""
        def __init__(self):
            import asyncio
            self._lock = asyncio.Lock()
            self._owner = None

        async def acquire(self, owner_id: str):
            """Blocking acquire (wait until free)."""
            await self._lock.acquire()
            self._owner = owner_id
            return True

        async def release(self, owner_id: str):
            if self._owner != owner_id:
                return False

            if self._lock.locked():
                self._lock.release()
                self._owner = None
            return True

        async def owner(self):
            """Return current lock owner for debug."""
            return self._owner


    def _get_discovery_lock_actor() -> "DiscoveryLock":
        """Return a global DiscoveryLock actor, creating it if missing."""
        name = "cluster_gpu_discovery_lock"
        namespace = "cluster_gpu_discovery"

        try:
            return ray.get_actor(name, namespace=namespace)
        except ValueError:
            return DiscoveryLock.options(
                name=name,
                namespace=namespace,
                lifetime="detached",
            ).remote()


    # ---------------------------------
    # Context manager for cluster lock
    # ---------------------------------
    class ClusterLockContext:
        """Synchronous context manager for DiscoveryLock actor."""
        def __init__(self, actor: "DiscoveryLock", verbose: bool = False):
            import uuid
            self._actor = actor
            self._verbose = verbose
            # Unique per-process token
            self._owner_id = f"{uuid.uuid4()}"

        def _ts(self) -> str:
            """Return timestamp with milliseconds."""
            from datetime import datetime
            return datetime.now().strftime("%H:%M:%S.%f")[:-3]

        def __enter__(self):
            # Block until the actor lock is acquired
            ray.get(self._actor.acquire.remote(self._owner_id))
            owner = ray.get(self._actor.owner.remote())
            if self._verbose:
                print(f"{self._ts()} [Cluster Lock] Acquired by {owner}")
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            result = ray.get(self._actor.release.remote(self._owner_id))
            if self._verbose:
                if result:
                    print(f"{self._ts()} [Cluster Lock] Released by {self._owner_id}")
                else:
                    print(f"{self._ts()} [Cluster Lock] Not Released -- not Owner")

            # don't suppress exceptions
            return False

except ImportError:
    pass


def discover_gpus(verbose: bool = False, no_cache: bool = False, max_local_discovery: int = 1, local_gpu_only: bool = False):
    """Discover local and cluster GPUs and measure FP8/FP4 GFLOPs."""
    # Step 1: get exclusive file lock
    try:
        with SimpleCrashSafeLock(verbose):

            # Step 2: refresh nvml_info before import torch
            refresh_nvml_info()

            # Step 3: check GPUs availability
            import torch
            if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
                if verbose:
                    print("No GPU is found")
                return {}

            # Step 4: delete cache if no_cache is True
            if no_cache:
                from .cache import delete_cache
                deleted = delete_cache()
                if verbose:
                    if deleted:
                        print("GPUs Cache is deleted\n")
                    else:
                        print("GPUs Cache does Not exist\n")

            # Step 5: Get local GPUs
            local_gpus = []
            for attempt in range(max_local_discovery):
                if verbose:
                    print(f"local GPUs Discovery Attempt : {attempt + 1} \n")
                local_gpus = discover_local_gpus(verbose)

            if local_gpu_only:
                result = {"local": local_gpus}

                from .cache import save_cache
                save_cache(local_gpus)

                if verbose:
                    import json
                    print("Save local GPUs info to cache")
                    print(json.dumps(local_gpus, indent=2))
                    print()

                return result

            # Step 6: Check if Ray is running and get cluster GPUs
            cluster_gpus = []
            try:
                from .discover_cluster_gpus import discover_cluster_gpus

                import ray
                if ray.is_initialized():
                    lock_actor = _get_discovery_lock_actor()
                    with ClusterLockContext(lock_actor, verbose=verbose):
                        cluster_gpus = discover_cluster_gpus(verbose)
                else:
                    if verbose:
                        ray.init(address="auto")
                    else:
                        import logging
                        ray.init(address="auto", logging_level=logging.WARNING)

                    lock_actor = _get_discovery_lock_actor()
                    with ClusterLockContext(lock_actor, verbose=verbose):
                        cluster_gpus = discover_cluster_gpus(verbose)

                    ray.shutdown()
            except Exception as e:
                if verbose:
                    print(f"discover cluster GPUs Exception : {e}")
                pass

            # Step 7: Combine results
            result = {"local": local_gpus}

            # Only include "ray" key if there are cluster GPUs
            if cluster_gpus:
                result["ray"] = cluster_gpus
            else:
                from .cache import save_cache
                save_cache(local_gpus)

                if verbose:
                    import json
                    print("Save local GPUs info to cache")
                    print(json.dumps(local_gpus, indent=2))
                    print()

            return result
    except Exception as e:
        if verbose:
            print(f"Failed to acquire SimpleCrashSafeLock : {e}")

        return {"FAIL": str(e)}
