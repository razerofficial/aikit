"""Pytest configuration for functional tests.

These tests validate actual functionality of the aikit components,
mirroring the notebook workflows.
"""
import pytest
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session", autouse=True)
def warm_gpu_discovery():
    """Populate the GPU discovery cache before any `rzr-aikit` subprocess runs.

    In the shipped image a login shell pre-warms this in the background via
    /etc/profile.d/discover_local_gpus.sh. Tests run under `--entrypoint pytest`,
    which never sources /etc/profile.d, so the first `rzr-aikit model info` call
    would instead perform a full synchronous discovery (~40s) and blow past the
    30s subprocess timeouts in this suite. Warming it once here keeps the CLI
    tests deterministic and mirrors the runtime environment.
    """
    import subprocess

    Path.home().joinpath(".cache").mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["gpu-discovery", "--max-local-discovery", "1", "--local-gpu-only"],
            capture_output=True,
            timeout=180,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # No GPU tooling available; GPU-dependent tests skip on their own.
        pass


@pytest.fixture(scope="session")
def gpu_available():
    """Check if GPU is available for testing."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


@pytest.fixture(scope="session")
def skip_if_no_gpu(gpu_available):
    """Skip test if no GPU is available."""
    if not gpu_available:
        pytest.skip("GPU not available")


@pytest.fixture(scope="session")
def test_model():
    """Small model for testing LLM functionality."""
    return "Qwen/Qwen3.5-0.8B"


@pytest.fixture(scope="session")
def vllm_server_timeout():
    """Timeout for vLLM server operations in seconds."""
    return 120


@pytest.fixture
def cleanup_containers():
    """Cleanup Docker containers after tests."""
    import subprocess

    yield

    # Cleanup any test containers
    try:
        subprocess.run(
            ["docker", "ps", "-a", "--filter", "name=test-", "-q"],
            capture_output=True,
            check=False
        )
    except Exception:
        pass
