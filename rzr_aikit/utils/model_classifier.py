import os
import json
from enum import Enum, auto
from huggingface_hub import try_to_load_from_cache


class ModelCategory(Enum):
    BAGEL = auto()
    DIFFUSION = auto()
    STANDARD = auto()


def _read_json_from_model(model_name: str, filename: str) -> dict | None:
    if os.path.isdir(model_name):
        p = os.path.join(model_name, filename)
        if os.path.isfile(p):
            with open(p) as f:
                return json.load(f)
        return None
    path = try_to_load_from_cache(model_name, filename)
    if path and os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return None

# Sourced from vllm_omni.diffusion.utils/hf_utils.py
def _looks_like_bagel(model_name: str) -> bool:
    try:
        cfg = _read_json_from_model(model_name, "config.json")
        if not cfg:
            return False
        return cfg.get("model_type") == "bagel" or "BagelForConditionalGeneration" in (
            cfg.get("architectures") or []
        )
    except Exception:
        return False

def _is_diffusion(model_name: str) -> bool:
    try:
        idx = _read_json_from_model(model_name, "model_index.json")
        if idx is not None:
            return bool(idx.get("_class_name") and idx.get("_diffusers_version"))
        # Fallback: check hub file list without downloading
        try:
            from huggingface_hub import HfApi
            siblings = HfApi().model_info(model_name).siblings or []
            return any(s.rfilename == "model_index.json" for s in siblings)
        except Exception:
            return False
    except Exception:
        return False


def classify_model(model_name: str) -> ModelCategory:
    """Classify a model into a category to determine which fetcher to use.

    Checks in priority order:
    1. Bagel — config.json-based diffusion model (must be checked before _is_diffusion,
       which also returns True for Bagel but the DiffusionModelInfoFetcher cannot handle it)
    2. Diffusion — standard diffusers pipeline with model_index.json
    3. Standard — LLM/VLM with config.json
    """
    if _looks_like_bagel(model_name):
        return ModelCategory.BAGEL
    if _is_diffusion(model_name):
        return ModelCategory.DIFFUSION
    return ModelCategory.STANDARD
