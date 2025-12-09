import pytest
import os
import sys
from pathlib import Path

print("conftest.py LOADED")


# Add the project root directory to Python path (so both src and util are importable)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(autouse=True)
def setup_test_env():
    """Set up test environment."""
    # Disable VLLM logging for tests
    os.environ["VLLM_CONFIGURE_LOGGING"] = "0"
    
    # Set test cache directory
    os.environ["HF_HOME"] = "/tmp/test_hf_cache"
    
    yield
    
    # Cleanup
    if "VLLM_CONFIGURE_LOGGING" in os.environ:
        del os.environ["VLLM_CONFIGURE_LOGGING"]
    if "HF_HOME" in os.environ:
        del os.environ["HF_HOME"]