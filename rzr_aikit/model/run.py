import typer
from typing_extensions import Annotated
from rich.live import Live
from rich.table import Table
from rich.console import Console
from rich.panel import Panel
from rich.bar import Bar
import os

from rzr_aikit import model_app
from rzr_aikit.utils.spinner import spinner

console = Console()
SYSTEMONE_LOG = "/tmp/rzr_aikit_systemone.log"


class GPUStatus:
    def __init__(self, refresh_interval=1):
        """
        Initialize the GPU status display with a custom refresh interval.

        Args:
            refresh_interval (int): How often to refresh GPU info in seconds
        """
        self.refresh_interval = refresh_interval
        self.last_refresh = 0
        self.cached_infos = None
        import time

        self.time = time

    def __rich__(self) -> Panel:
        from rzr_aikit.utils.mlib import get_cuda_gpu_infos
        from accelerate.utils import convert_bytes

        current_time = self.time.time()
        if (
            self.cached_infos is None
            or (current_time - self.last_refresh) >= self.refresh_interval
        ):
            self.cached_infos = list(get_cuda_gpu_infos())
            self.last_refresh = current_time

        table = Table(show_header=True, box=None, padding=(0, 1))
        table.add_column("Index")
        table.add_column("Name")
        table.add_column("Memory Usage")
        table.add_column("")
        table.add_column("")
        table.add_row("", "", "", "", "")  # blank row for spacing
        for idx, info in enumerate(self.cached_infos):
            mem = info["mem"]
            usage = mem.used / mem.total
            bar = Bar(size=1, begin=0, end=usage, width=20, bgcolor="bright_black")
            table.add_row(
                str(idx),
                info["name"],
                f"{convert_bytes(mem.used)}/{convert_bytes(mem.total)}",
                bar,
                f"{mem.used / mem.total * 100:.2f}%",
            )
        return Panel(
            table,
            title=f"GPU Resource Utilisation",
            border_style="green",
        )


@model_app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=True,
)
def run(
    ctx: typer.Context,
    model: Annotated[
        str,
        typer.Argument(
            # This metavar tells Typer how to display the argument in the usage string.
            # This is the key to solving your problem.
            metavar="model [VLLM_SERVE_EXTRA_ARGS...]",
            help="The name of the model to run, followed by any extra arguments for the VLLM server.",
        ),
    ],
    omni: Annotated[
        bool,
        typer.Option(
            "--omni",
            help="Diffusion models require this flag to be set.",
            is_flag=True,
        ),
    ] = False,
    port: Annotated[
        int,
        typer.Option(
            "--port",
            "-p",
            help="Port to run the server on.",
        ),
    ] = 8000,
    systemone_port: Annotated[
        int,
        typer.Option(
            "--systemone-port",
            help="Port of the SystemOne API, started next to vLLM for decision models.",
        ),
    ] = 8001,
):
    """
    Run a model for inferencing.

    This command starts a vLLM inference server for the specified model. The server
    will automatically optimize parameters based on available GPU resources. You can
    pass additional vLLM server arguments after the model name to customize the
    configuration. Real-time GPU utilization is displayed during startup (except in
    Jupyter environments).

    Examples:

        $ rzr-aikit model run deepseek-ai/DeepSeek-R1-Distill-Llama-8B
        $ rzr-aikit model run Qwen/Qwen3.5-0.8B --max-model-len 4096
        $ rzr-aikit model run facebook/opt-125m --quantization bitsandbytes
    """
    flags: list[str] = ctx.args

    console.print(f"model : {model}", style="green", highlight=False)
    console.print(f"flags : {flags}", style="yellow")

    gpu = GPUStatus()
    if "JPY_PARENT_PID" in os.environ:
        engine = run_model(ctx, model, omni)
        if engine:
            start_server(engine, port, systemone_port)
    else:
        with Live(gpu, console=console, refresh_per_second=8, transient=True):
            engine = run_model(ctx, model, omni)
            if engine:
                start_server(engine, port, systemone_port)


@spinner("Optimizing parameters...", console=console)
def run_model(ctx: typer.Context, model: str, omni: bool):
    try:
        flags: list[str] = ctx.args

        from rzr_aikit.utils.SmartVLLMServe import SmartVLLMServe
        from rzr_aikit.utils.ModelInfoFetcher import ModelInfoFetcher
        from rzr_aikit.utils.connectivity import check_huggingface_connectivity

        has_connection = check_huggingface_connectivity()
        if not has_connection:
            flags.extend(["--served-model-name", model])
            fetcher = ModelInfoFetcher(model, allow_internet=has_connection)
            model = fetcher.model_path

        from rzr_aikit.utils.model_classifier import ModelCategory, classify_model, systemone_provider

        provider = systemone_provider(model) if classify_model(model) == ModelCategory.SYSTEMONE else None
        server_flags: list[str] = []
        if provider:
            # --lora-modules is a server flag, not an EngineArgs field: it goes straight to `vllm serve`
            provider_flags = systemone_vllm_flags(provider, model, flags)
            if "--lora-modules" in provider_flags:
                i = provider_flags.index("--lora-modules")
                server_flags, provider_flags = provider_flags[i : i + 2], provider_flags[:i] + provider_flags[i + 2 :]
            flags = flags + provider_flags
        engine = SmartVLLMServe(model, *flags)
        engine.server_flags = server_flags
        engine.system_one = provider is not None
        engine.systemone_provider = provider

        if omni:
            engine.omni = True
            return engine
        if hasattr(engine.engine_args, "max_num_seqs"):
            if engine.engine_args.max_num_seqs is None:
                engine.engine_args.max_num_seqs = 5

        try:
            engine.determine_dtype()
        except Exception as e:
            pass

        # manually suppress for now
        # print(engine.get_cli_args())
        # print()
        # print(engine.get_cli_string())

        max_attempts = 10
        exit_code = -1  # Initialize with a non-success code

        for attempt in range(max_attempts):
            console.print(
                f"--- Running command: Attempt {attempt + 1}/{max_attempts} ---"
            )

            # Execute the function that runs the subprocess

            exit_code = engine.get_max_model_len()
            # print(f"Attempt finished with exit code: {exit_code}")

            # If the exit code is 200, it was successful. Break the loop.
            if exit_code == -1 or exit_code == 255:
                console.print(f"\nFatal Error! Received exit code {exit_code}.")
                break
            elif exit_code == 200:
                console.print("\nSuccess! Received exit code 200.")
                break
            else:
                console.print("Attempt failed. Retrying...\n")
                # Optional: Add a small delay between retries
                # import time
                # time.sleep(2)
        else:
            # This 'else' block runs only if the 'for' loop completes without a 'break'
            console.print(f"\nCommand failed after {max_attempts} attempts. Giving up.")

        if exit_code == 200:
            return engine
    except Exception as e:
        console.print(f"Unexpected error: {e}")


def systemone_vllm_flags(provider: str, model: str, user_flags: list) -> list:
    """Extra `vllm serve` flags a provider needs. AutoTrust serves its decision head as the LoRA module
    `jev-decision`; the multimodal class (JEV-27B-VL) also needs at most 8 sequences per batch, or the
    LoRA path returns wrong probabilities."""
    if provider != "autotrust":
        return []
    import json
    import os

    from huggingface_hub import snapshot_download

    adapter = os.path.join(
        model if os.path.isdir(model) else snapshot_download(model, allow_patterns=["adapter_vllm/*", "calibration.json", "config.json"]),
        "adapter_vllm",
    )
    flags = [
        "--enable-lora", "--max-lora-rank", "32", "--lora-modules", f"jev-decision={adapter}",
        "--logprobs-mode", "processed_logprobs",
    ]
    cfg_path = os.path.join(os.path.dirname(adapter), "config.json")
    with open(cfg_path, encoding="utf-8") as f:
        archs = json.load(f).get("architectures") or []
    if any(a.endswith("ForConditionalGeneration") for a in archs):
        flags += ["--mamba-cache-mode", "align"]
        if "--max-num-seqs" not in user_flags:
            flags += ["--max-num-seqs", "8"]
    return flags


def start_systemone(engine, vllm_port: int, systemone_port: int):
    """Start the SystemOne wrapper next to vLLM, which keeps serving its own API untouched."""
    import subprocess
    import sys

    model = engine.engine_args.model
    served = engine.engine_args.served_model_name or model
    served = served if isinstance(served, str) else served[0]
    cmd = [
        sys.executable, "-m", "vllm_plugins.systemone.server", "--model", model,
        "--provider", engine.systemone_provider,
        "--served-model-name", served,
        "--vllm-url", f"http://localhost:{vllm_port}", "--port", str(systemone_port),
    ]
    return subprocess.Popen(
        cmd, stdout=open(SYSTEMONE_LOG, "w"), stderr=subprocess.STDOUT, start_new_session=True
    )


def wait_for_systemone(wrapper, port: int, timeout: float = 180):
    """Block until the wrapper answers /health; fail with its log if it exits first."""
    import time
    import urllib.request

    deadline = time.time() + timeout
    while time.time() < deadline:
        if wrapper.poll() is not None:
            with open(SYSTEMONE_LOG) as f:
                raise RuntimeError(f"SystemOne API failed to start:\n{''.join(f.readlines()[-10:])}")
        try:
            urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2)
            return
        except OSError:
            time.sleep(1)
    raise RuntimeError(f"SystemOne API did not answer on port {port} within {timeout:.0f}s; see {SYSTEMONE_LOG}")


@spinner("Starting up model server...", console=console)
def start_server(engine, port=None, systemone_port=8001):
    """
    Start the VLLM server with the given engine.
    """
    try:
        import re

        route_regex = re.compile(r"INFO\s+\S+:\d+:\s*(.*)")
        start_regex = re.compile(r"INFO:\s*(.*)")

        extra_args = ["--omni"] if getattr(engine, "omni", False) else None
        system_one = getattr(engine, "system_one", False)
        if system_one and (port or 8000) == systemone_port:
            raise ValueError(f"--systemone-port {systemone_port} is the vLLM port; pick another")
        server_flags = engine.server_flags if isinstance(getattr(engine, "server_flags", None), list) else []
        process = engine.serve(port=port, extra=(extra_args or []) + server_flags or None)
        start_print = False
        error_msg = None
        with open("/tmp/rzr_aikit.log", "r") as log_file:
            while True:
                piped_output = log_file.readline()
                if "Max retries exceeded" in piped_output:
                    raise Exception("Unable to establish connection to huggingface.co")
                # capture the first error message shown
                if "Error" in piped_output and error_msg is None:
                    error_msg = piped_output
                if not piped_output and process.poll() is not None:
                    raise Exception(error_msg)
                if "Starting vLLM API server" in piped_output:
                    start_print = True
                if start_print:
                    extracted = route_regex.search(piped_output)
                    if not extracted:
                        extracted = start_regex.search(piped_output)
                    console.print(extracted.group(1))
                if "Application startup complete" in piped_output:
                    if system_one:
                        wrapper = start_systemone(engine, port or 8000, systemone_port)
                        wait_for_systemone(wrapper, systemone_port)
                        console.print(
                            f"\nSystemOne API ready: POST http://localhost:{systemone_port}/v1/systemone"
                        )
                    else:
                        console.print(
                            "\nPlease use 'rzr-aikit model generate' to start generating responses."
                        )
                    return process
    except Exception as e:
        console.print(f"Error starting server: {e}")
        return None
