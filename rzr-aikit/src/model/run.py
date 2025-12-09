import typer
from typing_extensions import Annotated
from rich.live import Live
from rich.table import Table
from rich.console import Console
from rich.panel import Panel
from rich.bar import Bar
import os

from src import model_app
from util.spinner import spinner


console = Console()


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
        from util.mlib import get_cuda_gpu_infos
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
        $ rzr-aikit model run Qwen/Qwen3-0.6B --max-model-len 4096
        $ rzr-aikit model run facebook/opt-125m --quantization bitsandbytes
    """
    flags: list[str] = ctx.args

    console.print(f"model : {model}", style="green", highlight=False)
    console.print(f"flags : {flags}", style="yellow")

    gpu = GPUStatus()
    if "JPY_PARENT_PID" in os.environ:
        engine = run_model(ctx, model)
        if engine:
            start_server(engine)
    else:
        with Live(gpu, console=console, refresh_per_second=8, transient=True):
            engine = run_model(ctx, model)
            if engine:
                start_server(engine)


@spinner("Optimizing parameters...", console=console)
def run_model(ctx: typer.Context, model: str):
    try:
        flags: list[str] = ctx.args

        from util.SmartVLLMServe import SmartVLLMServe
        from util.ModelInfoFetcher import ModelInfoFetcher
        from util.connectivity import check_huggingface_connectivity

        has_connection = check_huggingface_connectivity()
        if not has_connection:
            flags.extend(["--served-model-name", model])
            fetcher = ModelInfoFetcher(model, allow_internet=has_connection)
            model = fetcher.model_path

        engine = SmartVLLMServe(model, *flags)

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


@spinner("Starting up model server...", console=console)
def start_server(engine):
    """
    Start the VLLM server with the given engine.
    """
    try:
        process = engine.serve()
        start_print = False
        while True:
            piped_output = process.stdout.readline()
            if not piped_output and piped_output.poll() is not None:
                break
            if "Starting vLLM API server" in piped_output:
                start_print = True
            if start_print:
                console.print(piped_output, end="")
            if "Application startup complete" in piped_output:
                console.print(
                    "\nPlease use 'rzr-aikit model generate' to start generating responses."
                )
                return process
    except Exception as e:
        console.print(f"Error starting server: {e}")
        return None
