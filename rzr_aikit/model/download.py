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
        from rzr_aikit.model.list import has_weights
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
    from rzr_aikit.utils.model_classifier import ModelCategory, classify_model

    with console.status("Starting model download..."):
        category = classify_model(model_name)

    try:
        if category == ModelCategory.STANDARD or category == ModelCategory.BAGEL:
            with console.status("Attempting download with vLLM loader..."):
                from vllm.config import LoadConfig, ModelConfig
                from vllm.model_executor.model_loader import get_model_loader

            model_config = ModelConfig(
                model=model_name,
                tokenizer=model_name,
                tokenizer_mode="auto",
                trust_remote_code=False,
                dtype="auto",
                seed=0,
            )
            load_config = LoadConfig(load_format="auto")
            loader = get_model_loader(load_config)
            loader.download_model(model_config)
        else:
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
