from src import model_app
from util.spinner import spinner

from rich.console import Console
from rich.box import ROUNDED

console = Console()


@model_app.command()
@spinner("Getting locally cached models...", console=console)
def list():
    """
    List all locally cached models.

    This command displays a table of all models currently stored in the local cache.
    For each model, it shows the model name, size on disk, and compatibility status
    for both local and distributed inference. Compatibility is determined by comparing
    model memory requirements against available GPU memory.

    Compatibility indicators:

        ✅ Yes      - Model can run natively with full precision
        🟡 Limited  - Model can run with 4-bit quantization (reduced accuracy)
        ❌ No       - Insufficient memory to run the model
        -           - Distributed compatibility unavailable (not connected to Ray cluster)

    """

    try:
        from util.mlib import get_cuda_total_vram, check_model_fit
        from util.ModelInfoFetcher import ModelInfoFetcher
        from huggingface_hub import scan_cache_dir

        from rich.table import Table
        from humanize import naturalsize
        import os

        os.environ["VLLM_CONFIGURE_LOGGING"] = "0"

        table = Table(
            box=ROUNDED,
            show_header=True,
            header_style="bold",
            expand=True,
            border_style="green",
        )
        table.add_column("Model Name", overflow="fold", ratio=3)
        table.add_column("Size", overflow="ellipsis", ratio=1)
        table.add_column("Compatible (Local)", overflow="ellipsis", ratio=1)
        table.add_column("Compatible (Dist.)", overflow="ellipsis", ratio=1)

        local_memory = get_cuda_total_vram()
        dist_memory = get_cuda_total_vram(distributed=True)

        models = []
        info = scan_cache_dir()

        for repo in info.repos:
            if repo.repo_type != "model":
                continue
            if not has_weights(repo):
                continue
            try:
                fetcher = ModelInfoFetcher(repo.repo_id, allow_internet=False)
                weight_size = fetcher.total_bytes
                data_type = fetcher.dtype
                quant_type = None
                if fetcher.quant_config:
                    quant_type = fetcher.quant_config["quant_method"]

                local = check_model_fit(
                    local_memory, weight_size, quant_type if quant_type else data_type
                )

                distributed = "-"
                if dist_memory > 0:
                    distributed = check_model_fit(
                        dist_memory,
                        weight_size,
                        quant_type if quant_type else data_type,
                    )
                models.append((repo.repo_id, weight_size, local, distributed))
            except Exception as e:
                console.print(f"[yellow]Skipping model[/]  '{repo.repo_id}': {e}")

        models.sort(key=lambda x: x[0])
        inflight_used = False

        for name, size, local, distributed in models:
            table.add_row(
                name,
                naturalsize(size),
                local,
                distributed,
            )
            if local == "🟡 Limited" or distributed == "🟡 Limited":
                inflight_used = True

        console.print(table)
        if inflight_used:
            console.print(
                "🟡 The model supports 4-bit on-the-fly quantization using '--quantization bitsandbytes', with reduced accuracy."
            )

        if dist_memory == 0:
            console.print(
                "[yellow]Not connected to Ray cluster, distributed compatibility skipped.[/]"
            )
        if dist_memory == -1:
            console.print(
                "[yellow]GPU discovery is in progress, distributed compatibility skipped.[/]"
            )
    except Exception as e:
        console.print(e)


def has_weights(repo):
    """
    Return True if the repo has any file that looks like a model weight file.
    Tries snapshots, then falls back to scanning the HF cache path directly.
    """
    import os

    weight_exts = (".bin", ".safetensors", ".pt")

    # 1. Try metadata-based check (snapshots)
    snapshots = getattr(repo, "snapshots", [])
    if snapshots:
        for snapshot in snapshots:
            files = getattr(snapshot, "files", [])
            for file_info in files:
                f = getattr(file_info, "file_name", None)
                if (
                    f
                    and f.endswith(weight_exts)
                    and not f.startswith(("config", "tokenizer"))
                ):
                    return True
        return False

    # 2. Fallback: scan the HuggingFace cache path directly
    repo_id = getattr(repo, "repo_id", None)
    if repo_id:
        cache_base = os.path.expanduser("~/.cache/huggingface/hub")
        repo_dir = os.path.join(cache_base, f"models--{repo_id.replace('/', '--')}")
        if os.path.isdir(repo_dir):
            for root, dirs, files in os.walk(repo_dir):
                for f in files:
                    if f.endswith(weight_exts) and not f.startswith(
                        ("config", "tokenizer")
                    ):
                        return True
    return False


if __name__ == "__main__":
    list()
