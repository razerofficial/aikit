from src import model_app
import typer
from typing_extensions import Annotated
from rich import print


@model_app.command()
def remove(model_name: Annotated[str, typer.Argument(help="organization/model_name")]):
    """
    Remove a model from the local cache.

    This command deletes the specified model and all its associated files from the
    local HuggingFace cache directory. An error message is displayed if the model is
    not found.

    Examples:

        $ rzr-aikit model remove microsoft/DialoGPT-small
        $ rzr-aikit model remove facebook/opt-125m
    """
    try:
        from huggingface_hub import scan_cache_dir

        info = scan_cache_dir()
        for repo in info.repos:
            if repo.repo_id == model_name:
                commit_hashes = [revision.commit_hash for revision in repo.revisions]
                delete_strategy = info.delete_revisions(*commit_hashes)
                delete_strategy.execute()
                print(f"Model '{model_name}' has been removed from the local cache.")
                return
        print(f"Model '{model_name}' not found.")
    except Exception as e:
        print(e)
