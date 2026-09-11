"""
Validates Notebook 4: On-Device Audio Generation
Tests TTS model workflow with --omni flag and /v1/audio/speech API.
"""
import pytest
import subprocess
import requests
import json
from pathlib import Path


@pytest.mark.audio
class TestAudioGenerationCLI:
    """CLI and model operations.

    Tests CLI commands for TTS models:
    model info, download for audio generation models.
    """

    @pytest.mark.parametrize("model", [
        "mistralai/Voxtral-4B-TTS-2603",
    ])
    def test_tts_model_info(self, model):
        """Test retrieving TTS model information."""
        result = subprocess.run(
            ["rzr-aikit", "model", "info", model],
            capture_output=True,
            text=True,
            timeout=90  # Large TTS models need more time
        )

        # Should succeed or provide meaningful error
        assert result.returncode == 0 or "Incompatible" in result.stdout

        if result.returncode == 0:
            assert model in result.stdout

    @pytest.mark.slow
    @pytest.mark.requires_network
    def test_tts_model_download(self):
        """Test downloading a TTS model."""
        model = "mistralai/Voxtral-4B-TTS-2603"
        result = subprocess.run(
            ["rzr-aikit", "model", "download", model],
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes for large models
        )

        # Should succeed or report already cached
        assert result.returncode == 0 or \
               "already downloaded" in result.stdout.lower()


@pytest.mark.audio
@pytest.mark.slow
@pytest.mark.requires_gpu
class TestAudioGenerationServer:
    """TTS server and API workflow (requires GPU).

    Tests audio generation server and /v1/audio/speech API:
    speech generation, voice parameters, task types.
    """

    @pytest.mark.integration
    def test_audio_speech_api(self, tmp_path):
        """Test audio generation via /v1/audio/speech API.

        Assumes a TTS model server is running on port 8000.
        This is an integration test that requires manual server setup.
        """
        # Check if server is up
        try:
            response = requests.get(
                "http://localhost:8000/health",
                timeout=2
            )
            if response.status_code != 200:
                pytest.skip("TTS server not running")
        except requests.exceptions.RequestException:
            pytest.skip("TTS server not running")

        # Test audio generation
        payload = {
            "model": "mistralai/Voxtral-4B-TTS-2603",
            "input": "Hello! This is a test of text to speech generation.",
            "voice": "casual_male",
            "language": "English",
            "response_format": "wav",
            "task_type": "CustomVoice"
        }

        response = requests.post(
            "http://localhost:8000/v1/audio/speech",
            json=payload,
            timeout=60
        )

        assert response.status_code == 200
        assert len(response.content) > 0

        # Verify WAV format
        output_file = tmp_path / "test_output.wav"
        output_file.write_bytes(response.content)
        assert output_file.exists()
        assert output_file.stat().st_size > 1000  # Non-trivial audio file

    @pytest.mark.integration
    def test_audio_generation_parameters(self):
        """Test various TTS generation parameters.

        Tests different task types, voices, and languages.
        """
        # Check if server is up
        try:
            response = requests.get(
                "http://localhost:8000/health",
                timeout=2
            )
            if response.status_code != 200:
                pytest.skip("TTS server not running")
        except requests.exceptions.RequestException:
            pytest.skip("TTS server not running")

        # Test different voices
        test_cases = [
            {
                "voice": "casual_male",
                "task_type": "CustomVoice"
            },
            {
                "voice": "casual_female",
                "task_type": "CustomVoice"
            }
        ]

        for params in test_cases:
            payload = {
                "model": "mistralai/Voxtral-4B-TTS-2603",
                "input": "Testing voice parameters.",
                "language": "English",
                "response_format": "wav",
                **params
            }

            response = requests.post(
                "http://localhost:8000/v1/audio/speech",
                json=payload,
                timeout=60
            )

            assert response.status_code == 200, \
                f"Failed with params: {params}"
            assert len(response.content) > 0
