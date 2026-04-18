#!/usr/bin/env python3
import typer
from typing_extensions import Annotated
from rich.console import Console
from rich.panel import Panel
from rich.bar import Bar

from rzr_aikit.utils.spinner import spinner

console = Console()


@spinner("Retrieving model performance and resource utilization...", console=console)
def metrics(
    ctx: typer.Context,
    help: Annotated[
        bool,
        typer.Option(
            "--help",
            help="Display help information including usage syntax, available options, and examples.",
        ),
    ] = False,
):
    """
    Retrieve model and hardware metrics.

    This command displays comprehensive performance metrics for running models,
    including time to first token (TTFT), output throughput for single and parallel
    requests, uptime, and GPU resource utilization. Metrics are collected from the
    vLLM server's Prometheus endpoints.

    """
    if help:
        typer.echo(ctx.get_help())
        raise typer.Exit()

    from collections import defaultdict
    from accelerate.utils import convert_bytes
    import time
    from rzr_aikit.utils.mlib import get_cuda_gpu_infos, get_metrics, get_running_models
    from rich import print
    from rich.table import Table

    stats = defaultdict(dict)  # {model_name: {ttft: }}
    for model in get_running_models():
        stats[model.id]
    metrics = get_metrics()
    for metric in metrics:
        if metric.name == "vllm:time_to_first_token_seconds":
            for ttft_sample in metric.samples:
                if ttft_sample.name == "vllm:time_to_first_token_seconds_count":
                    stats[ttft_sample.labels["model_name"]][
                        "ttft_count"
                    ] = ttft_sample.value
                elif ttft_sample.name == "vllm:time_to_first_token_seconds_sum":
                    stats[ttft_sample.labels["model_name"]][
                        "ttft_sum"
                    ] = ttft_sample.value
        elif metric.name == "vllm:request_success_created":
            # no dedicated metric for start time, use _created for it
            for time_sample in metric.samples:
                stats[time_sample.labels["model_name"]]["time_ts"] = time_sample.value
        elif metric.name == "vllm:time_per_output_token_seconds":
            for trpt_sample in metric.samples:
                if trpt_sample.name == "vllm:time_per_output_token_seconds_sum":
                    stats[trpt_sample.labels["model_name"]][
                        "trpt_sum"
                    ] = trpt_sample.value
                elif trpt_sample.name == "vllm:time_per_output_token_seconds_count":
                    stats[trpt_sample.labels["model_name"]][
                        "trpt_count"
                    ] = trpt_sample.value
        elif metric.name == "vllm:generation_tokens":
            sample = metric.samples[0]
            stats[sample.labels["model_name"]]["gtt"] = sample.value
        elif metric.name == "vllm:e2e_request_latency_seconds":
            samples = list(
                filter(
                    lambda s: s.name == "vllm:e2e_request_latency_seconds_bucket",
                    reversed(metric.samples),
                )
            )
            # get upper bound
            for sam1, sam2 in zip(samples, samples[1:]):
                if sam1.value > sam2.value:
                    # TODO Consider number of requests as a gactor
                    # stats[sam1.labels["model_name"]]["val"] = float(sam1.value)
                    stats[sam1.labels["model_name"]]["bound"] = float(sam1.labels["le"])
                    break
            if "bound" not in stats[sam1.labels["model_name"]]:
                stats[sam1.labels["model_name"]]["bound"] = float(
                    samples[-1].labels["le"]
                )

    table = Table(show_header=True, header_style="bold", box=None)
    table.add_column("Model Name", no_wrap=True)
    table.add_column("Uptime")
    table.add_column("TTFT")
    table.add_column("Output Throughput (single)")
    table.add_column("Output Throughput (parallel)")
    table.add_row("", "", "", "", "")  # blank row for spacing
    for model_name, model_stat in stats.items():
        # not pythonic but the condition gets too long
        try:
            time_de = time.time() - model_stat["time_ts"]
            time_str = time.strftime("%H:%M:%S", time.gmtime(time_de))
        except KeyError:
            time_str = None
        try:
            ttft_avg = model_stat["ttft_sum"] / model_stat["ttft_count"]
            ttft_str = f"{ttft_avg * 1000:.2f} ms"
        except (KeyError, ZeroDivisionError):
            ttft_str = None
        # this is statistically questionable but anyways
        try:
            token_per_sec = model_stat["trpt_count"] / model_stat["trpt_sum"]
            token_str = f"{token_per_sec:.2f} tokens/sec"
        except (KeyError, ZeroDivisionError):
            token_str = None
        try:
            token_per_sec_par = model_stat["gtt"] / model_stat["bound"]
            token_par_str = f"{token_per_sec_par:.2f} tokens/sec"
        except (KeyError, ZeroDivisionError):
            token_par_str = None
        table.add_row(model_name, time_str, ttft_str, token_str, token_par_str)

    perf_panel = Panel(
        table,
        title=f"Model Performance Summary",
        title_align="left",
        border_style="green",
        expand=False,
    )

    from psutil import virtual_memory

    ram_metric = virtual_memory()
    usage = (ram_metric.total - ram_metric.available) / ram_metric.total
    bar = Bar(size=1, begin=0, end=usage, width=20, bgcolor="bright_black")

    resource_table = Table(show_header=True, header_style="bold", box=None)
    resource_table.add_column("Name")
    resource_table.add_column("Usage")
    # blank row for spacing
    resource_table.add_row("", "", "", "", "")
    resource_table.add_row(
        "System RAM",
        f"{convert_bytes(ram_metric.total - ram_metric.available)}/{convert_bytes(ram_metric.total)}",
        bar,
        f"{ram_metric.percent}%",
    )

    gpu_infos = get_cuda_gpu_infos()
    for info in gpu_infos:
        mem = info["mem"]
        usage = mem.used / mem.total
        bar = Bar(size=1, begin=0, end=usage, width=20, bgcolor="bright_black")

        resource_table.add_row(
            info["name"],
            f"{convert_bytes(mem.used)}/{convert_bytes(mem.total)}",
            bar,
            f"{mem.used / mem.total * 100:.2f}%",
        )
    ram_panel = Panel(
        resource_table,
        title=f"Resource Utilization",
        title_align="left",
        border_style="green",
        expand=False,
    )

    console.print(perf_panel)
    console.print(ram_panel)
