from src import model_app
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

        $ rzr-aikit model download Qwen/Qwen3-0.6B
        $ rzr-aikit model download facebook/opt-125m
    """
    # Check if model is already downloaded
    is_downloaded = check_if_downloaded(model_name)
    if is_downloaded:
        console.print(f"Model '{model_name}' is already downloaded.")
        return

    download_model(model_name)


def check_if_downloaded(model_name: str):
    """Check if model is already downloaded locally."""
    try:
        from src.model.list import has_weights
        from huggingface_hub import scan_cache_dir

        info = scan_cache_dir()
        for repo in info.repos:
            if repo.repo_id == model_name and has_weights(repo):
                return True
        return False
    except Exception:
        return False


def download_model(model_name: str):
    """Download the model."""

    try:
        with console.status("Preparing download..."):
            from vllm.config import LoadConfig, ModelConfig
            from vllm.model_executor.model_loader import get_model_loader

        model_config = ModelConfig(
            model=model_name,
            task="auto",
            tokenizer=model_name,
            tokenizer_mode="auto",
            trust_remote_code=False,
            dtype="auto",
            seed=None,
        )
        load_config = LoadConfig(load_format="auto")
        loader = get_model_loader(load_config)
        loader.download_model(model_config)

        console.print(f"Model '{model_name}' downloaded successfully.")

    except Exception as e:
        from src.model.remove import remove

        console.print(e)
        console.print("Removing model from cache...")
        remove(model_name)


if __name__ == "__main__":
    download("facebook/opt-125m")
