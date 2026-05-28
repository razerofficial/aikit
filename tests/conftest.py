import pytest
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Mock vllm_omni and submodules so DiffusionModelInfoFetcher can be imported
# without vllm being installed in the test environment.
_vllm_omni_registry = MagicMock()
_vllm_omni_registry._DIFFUSION_MODELS = {}
_vllm_omni_utils_registry = MagicMock()
_vllm_omni_utils_registry.is_diffusion_model = MagicMock(return_value=False)
_vllm_omni_diffusion_utils_hf_utils = MagicMock()
_vllm_omni_diffusion_utils_hf_utils._looks_like_bagel = MagicMock(return_value=False)
_vllm_omni_diffusion_utils_hf_utils.is_diffusion_model = MagicMock(return_value=False)
sys.modules.setdefault("vllm_omni", MagicMock())
sys.modules.setdefault("vllm_omni.diffusion", MagicMock())
sys.modules.setdefault("vllm_omni.diffusion.registry", _vllm_omni_registry)
sys.modules.setdefault("vllm_omni.diffusion.utils", MagicMock())
sys.modules.setdefault("vllm_omni.diffusion.utils.hf_utils", _vllm_omni_diffusion_utils_hf_utils)
sys.modules.setdefault("vllm_omni.utils", MagicMock())
sys.modules.setdefault("vllm_omni.utils.registry", _vllm_omni_utils_registry)


# Add the project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(autouse=True)
def setup_test_env():
    """Set up test environment."""
    os.environ["VLLM_CONFIGURE_LOGGING"] = "0"
    os.environ["HF_HOME"] = "/tmp/test_hf_cache"

    yield

    if "VLLM_CONFIGURE_LOGGING" in os.environ:
        del os.environ["VLLM_CONFIGURE_LOGGING"]
    if "HF_HOME" in os.environ:
        del os.environ["HF_HOME"]
