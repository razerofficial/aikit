import os

from typing import Optional, Sequence

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError


def resolve_cached_path(
    model: str,
    ignore_patterns: Optional[Sequence[str]] = None,
) -> Optional[str]:
    """Return a usable local path for a model, or None if it isn't available offline.

    Checks for a local directory first, then the HuggingFace cache.

    `ignore_patterns` must match the patterns used at download time. Since
    huggingface_hub 1.26, `local_files_only=True` validates the cached snapshot
    against the repo's file listing and rejects it as incomplete, so a download
    that deliberately skipped files has to skip them here too.
    """
    if os.path.isdir(model):
        return model

    try:
        return snapshot_download(
            repo_id=model,
            local_files_only=True,
            ignore_patterns=list(ignore_patterns) if ignore_patterns else None,
        )
    except LocalEntryNotFoundError:
        return None
    except Exception:
        return None


def is_cached(model: str, ignore_patterns: Optional[Sequence[str]] = None) -> bool:
    """
    Renamed to be used in model/download.py
    """
    return resolve_cached_path(model, ignore_patterns) is not None
