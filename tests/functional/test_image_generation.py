"""
Validates Notebook 3: On-Device Image Generation
Tests diffusion model workflow with --omni flag and /v1/images/generations API.
"""
import pytest
import subprocess
import time
import requests
import json
from pathlib import Path


@pytest.mark.image
class TestImageGenerationCLI:
    """CLI and model operations.

    Tests CLI commands for diffusion models:
    model info, download, --omni flag validation, UI commands.
    """

    @pytest.mark.parametrize("model", [
        "Tongyi-MAI/Z-Image-Turbo",
    ])
    def test_diffusion_model_info(self, model):
        """Test retrieving diffusion model information."""
        result = subprocess.run(
            ["rzr-aikit", "model", "info", model],
            capture_output=True,
            text=True,
            timeout=90  # Large diffusion models need more time
        )

        # Should succeed or provide meaningful error
        assert result.returncode == 0 or "Incompatible" in result.stdout

        if result.returncode == 0:
            # Check for key metadata fields
            assert model in result.stdout
            assert any(keyword in result.stdout.lower()
                      for keyword in ["diffusion", "image", "type"])

    @pytest.mark.slow
    @pytest.mark.requires_network
    def test_diffusion_model_download(self):
        """Test downloading a diffusion model."""
        model = "Tongyi-MAI/Z-Image-Turbo"
        result = subprocess.run(
            ["rzr-aikit", "model", "download", model],
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes for large models
        )

        # Should succeed or report already cached
        assert result.returncode == 0 or \
               "already downloaded" in result.stdout.lower()

    def test_omni_flag_available(self):
        """Test that diffusion models support --omni flag."""
        result = subprocess.run(
            ["rzr-aikit", "model", "run", "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )

        assert result.returncode == 0
        assert "--omni" in result.stdout

    def test_ui_run_command_exists(self):
        """Test that rzr-aikit ui run command is available."""
        result = subprocess.run(
            ["rzr-aikit", "ui", "run", "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )

        assert result.returncode == 0

    def test_ui_stop_command_exists(self):
        """Test that rzr-aikit ui stop command is available."""
        result = subprocess.run(
            ["rzr-aikit", "ui", "stop", "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )

        assert result.returncode == 0


@pytest.mark.image
@pytest.mark.slow
@pytest.mark.requires_gpu
class TestImageGenerationServer:
    """Server and API workflow (requires GPU).

    Tests image generation server and API endpoint:
    /v1/images/generations with various parameters.
    """

    @pytest.mark.integration
    def test_image_generation_api(self):
        """Test image generation via API endpoint.

        Assumes a diffusion model server is running on port 8000.
        This is an integration test that requires manual server setup.
        """
        # Check if server is up
        try:
            response = requests.get(
                "http://localhost:8000/health",
                timeout=2
            )
            if response.status_code != 200:
                pytest.skip("Image generation server not running")
        except requests.exceptions.RequestException:
            pytest.skip("Image generation server not running")

        # Test image generation
        payload = {
            "model": "Tongyi-MAI/Z-Image-Turbo",
            "prompt": "A cat sitting on a keyboard",
            "n": 1,
            "size": "512x512"
        }

        response = requests.post(
            "http://localhost:8000/v1/images/generations",
            json=payload,
            timeout=60
        )

        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert len(data["data"]) > 0
