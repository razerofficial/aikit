from enum import Enum, auto


class ModelCategory(Enum):
    BAGEL = auto()
    DIFFUSION = auto()
    STANDARD = auto()


def classify_model(model_name: str) -> ModelCategory:
    """Classify a model into a category to determine which fetcher to use.

    Checks in priority order:
    1. Bagel — config.json-based diffusion model (must be checked before is_diffusion_model,
       which also returns True for Bagel but the DiffusionModelInfoFetcher cannot handle it)
    2. Diffusion — standard diffusers pipeline with model_index.json
    3. Standard — LLM/VLM with config.json
    4. Generic — fallback for TTS/voice/unknown formats
    """
    from vllm_omni.diffusion.utils.hf_utils import _looks_like_bagel, is_diffusion_model

    if _looks_like_bagel(model_name):
        return ModelCategory.BAGEL

    if is_diffusion_model(model_name):
        return ModelCategory.DIFFUSION

    return ModelCategory.STANDARD
