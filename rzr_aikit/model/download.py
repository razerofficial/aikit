from rzr_aikit import model_app
import typer
from typing_extensions import Annotated
import os
from rich.console import Console

os.environ["VLLM_CONFIGURE_LOGGING"] = "0"
console = Console()


@model_app.command()
def download(
    model_name: Annotated[
        str, typer.Argument(help="organization/model_name")
    ] = "facebook/opt-125m",
):
    """
    Download a model from HuggingFace.

    This command downloads the specified model from HuggingFace Hub and stores it
    in the local cache directory. The download includes model weights and associated
    configuration files. If the model is already cached locally, the command will
    skip the download and notify you.

    Examples:

        $ rzr-aikit model download Qwen/Qwen3.5-0.8B
        $ rzr-aikit model download facebook/opt-125m
    """
    from rzr_aikit.utils.hf_cache import is_cached

    if is_cached(model_name):
        console.print(f"Model '{model_name}' is already downloaded.")
        return

    download_model(model_name)


def download_model(model_name: str):
    """Download the model."""

    try:
        from huggingface_hub import snapshot_download

        with console.status("Downloading model snapshot..."):
            snapshot_download(repo_id=model_name, local_files_only=False)

        console.print(f"Model '{model_name}' downloaded successfully.")

    except Exception as download_error:
        try:
            from rzr_aikit.model.remove import remove

            console.print(download_error)
            console.print("Removing any partial downloads from cache...")
            remove(model_name)
        except Exception:
            pass


if __name__ == "__main__":
    download("facebook/opt-125m")
