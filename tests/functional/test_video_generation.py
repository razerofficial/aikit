"""
Validates Notebook 5: On-Device Video Generation
Tests text-to-video diffusion workflow with --omni flag and async /v1/videos API.
"""
import pytest
import subprocess
import requests
import json
import time
from pathlib import Path


@pytest.mark.video
class TestVideoGenerationCLI:
    """CLI and model operations.

    Tests CLI commands for video diffusion models:
    model info, download, --omni flag validation, UI commands.
    """

    @pytest.mark.parametrize("model", [
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    ])
    def test_video_model_info(self, model):
        """Test retrieving video diffusion model information."""
        result = subprocess.run(
            ["rzr-aikit", "model", "info", model],
            capture_output=True,
            text=True,
            timeout=90  # Large video models need more time
        )

        # Should succeed or provide meaningful error
        assert result.returncode == 0 or "Incompatible" in result.stdout

        if result.returncode == 0:
            assert model in result.stdout

    @pytest.mark.slow
    @pytest.mark.requires_network
    def test_video_model_download(self):
        """Test downloading a video diffusion model."""
        model = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
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
        """Test that video diffusion models support --omni flag."""
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


@pytest.mark.video
@pytest.mark.slow
@pytest.mark.requires_gpu
class TestVideoGenerationServer:
    """Async video generation workflow (requires GPU).

    Tests video generation server and async /v1/videos API:
    job submission, status polling, parameter variations.
    """

    @pytest.mark.integration
    def test_video_generation_api(self):
        """Test video generation via /v1/videos API.

        Assumes a video diffusion model server is running on port 8000.
        This is an integration test that requires manual server setup.
        """
        # Check if server is up
        try:
            response = requests.get(
                "http://localhost:8000/health",
                timeout=2
            )
            if response.status_code != 200:
                pytest.skip("Video generation server not running")
        except requests.exceptions.RequestException:
            pytest.skip("Video generation server not running")

        # Test video generation request
        files = {
            'prompt': (None, 'A cat walking on grass'),
            'width': (None, '480'),
            'height': (None, '320'),
            'fps': (None, '8'),
            'num_frames': (None, '16'),  # Short video for testing
            'negative_prompt': (None, 'low quality, blurry')
        }

        response = requests.post(
            "http://localhost:8000/v1/videos",
            files=files,
            timeout=10
        )

        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        video_id = data["id"]

        # Test status check
        status_response = requests.get(
            f"http://localhost:8000/v1/videos/{video_id}",
            timeout=5
        )

        assert status_response.status_code == 200
        status_data = status_response.json()
        assert "status" in status_data
        # Must match vllm_omni's VideoGenerationStatus enum
        # (vllm_omni/entrypoints/openai/protocol/videos.py), which follows the
        # OpenAI Videos API vocabulary. A freshly submitted job reports
        # "queued" or "in_progress", so those have to be accepted here.
        assert status_data["status"] in ["queued", "in_progress", "completed", "failed"]

    @pytest.mark.integration
    def test_video_generation_parameters(self):
        """Test various video generation parameters.

        Tests different resolutions and frame counts.
        """
        # Check if server is up
        try:
            response = requests.get(
                "http://localhost:8000/health",
                timeout=2
            )
            if response.status_code != 200:
                pytest.skip("Video generation server not running")
        except requests.exceptions.RequestException:
            pytest.skip("Video generation server not running")

        # Test different parameter combinations
        test_cases = [
            {"width": "480", "height": "320", "fps": "8", "num_frames": "16"},
            {"width": "720", "height": "480", "fps": "12", "num_frames": "24"},
        ]

        for params in test_cases:
            files = {
                'prompt': (None, 'Test video generation'),
                **{k: (None, v) for k, v in params.items()}
            }

            response = requests.post(
                "http://localhost:8000/v1/videos",
                files=files,
                timeout=10
            )

            assert response.status_code == 200, \
                f"Failed with params: {params}"
            assert "id" in response.json()
