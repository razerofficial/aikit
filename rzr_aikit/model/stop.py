from rich import print
import psutil


from rzr_aikit import model_app

# A unique string to identify the VLLM process.
# VLLM is often launched with a command containing "vllm.entrypoints",
# making this a reliable identifier.
VLLM_IDENTIFIER = "bin/vllm"


def find_vllm_processes():
    """
    Finds all running processes that match the VLLM_IDENTIFIER.

    Returns:
        A list of psutil.Process objects matching the identifier.
    """
    found_processes = []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            # Check if the process's command line contains the identifier
            if process.info["cmdline"] and any(
                VLLM_IDENTIFIER in arg for arg in process.info["cmdline"]
            ):
                found_processes.append(process)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            # Some processes might be gone or inaccessible, just skip them
            pass
    return found_processes


def stop_process(process: psutil.Process):
    """
    Stops a given process gracefully, with a fallback to a force kill.
    """
    try:
        pid = process.pid
        print(f"--> Found VLLM process with PID: {pid}")
        # print(f"    Command: {' '.join(process.cmdline())}")

        # 1. Try to terminate gracefully (sends SIGTERM)
        print(f"--> Attempting to terminate process {pid} gracefully...")

        import subprocess
        import os
        import contextlib

        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(
                devnull
            ):
                # subprocess.run(
                #     ["bash", "-c", f"exec {pid} 1>/dev/null 2>/dev/null || true"],
                #     stdout=subprocess.DEVNULL,
                #     stderr=subprocess.DEVNULL,
                #     timeout=1,
                # )
                process.terminate()

                # 2. Wait for the process to die
                process.wait(timeout=15)  # Wait up to 15 seconds

        print(f"--> Process {pid} stopped successfully.")

    except psutil.TimeoutExpired:
        # 3. If it's still alive, force kill it (sends SIGKILL)
        print(f"!! Process {pid} did not terminate gracefully. Forcing kill...")
        process.kill()
        process.wait()  # Wait for the kill to complete
        print(f"--> Process {pid} killed.")
    except psutil.NoSuchProcess:
        # The process was already gone
        print(f"--> Process {pid} was already stopped.")
    except Exception as e:
        print(f"!! An error occurred while trying to stop process {pid}: {e}")


@model_app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=True,
)
def stop():
    """
    Terminate a running model inference server.

    This command searches for and stops all running vLLM server processes. It
    attempts a graceful shutdown first (SIGTERM), waiting up to 15 seconds. If the
    process doesn't terminate gracefully, it will force kill (SIGKILL). If multiple
    vLLM servers are running, all will be stopped sequentially.

    Examples:

        $ rzr-aikit model stop
    """
    print("Searching for running VLLM processes...")
    vllm_processes = find_vllm_processes()

    if not vllm_processes:
        print("No running VLLM processes found.")
    elif len(vllm_processes) == 1:
        stop_process(vllm_processes[0])
    else:
        print(f"!! Found {len(vllm_processes)} VLLM processes. stop them one by one")
        for proc in vllm_processes:
            try:
                print(f"  - PID: {proc.pid}")
                stop_process(proc)
            except Exception:
                pass
