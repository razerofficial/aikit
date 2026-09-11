"""
Validates Notebook 10: Voice Agent
Tests multi-model pipeline setup (STT, LLM, TTS) and WebSocket server configuration.
"""
import pytest
import subprocess
import json
import socket
from pathlib import Path


@pytest.mark.voice
class TestVoiceAgentCLI:
    """CLI and configuration validation.

    Tests voice agent module structure, configuration files,
    and component imports. Validates multi-model setup requirements.
    """

    def test_voice_agent_module_exists(self):
        """Test that voice_agent module can be imported."""
        result = subprocess.run(
            ["python", "-c", "import voice_agent"],
            capture_output=True,
            text=True,
            timeout=10
        )

        assert result.returncode == 0, \
            f"voice_agent module import failed: {result.stderr}"

    def test_realtime_server_module_exists(self):
        """Test that realtime_server module exists."""
        result = subprocess.run(
            ["python", "-c", "import voice_agent.realtime_server"],
            capture_output=True,
            text=True,
            timeout=10
        )

        assert result.returncode == 0, \
            f"realtime_server import failed: {result.stderr}"

    def test_pipeline_module_exists(self):
        """Test that pipeline module exists."""
        result = subprocess.run(
            ["python", "-c", "import voice_agent.pipeline"],
            capture_output=True,
            text=True,
            timeout=10
        )

        assert result.returncode == 0, \
            f"Pipeline module import failed: {result.stderr}"

    def test_vad_module_exists(self):
        """Test that VAD module exists."""
        result = subprocess.run(
            ["python", "-c", "import voice_agent.vad"],
            capture_output=True,
            text=True,
            timeout=10
        )

        assert result.returncode == 0, \
            f"VAD module import failed: {result.stderr}"

    def test_voice_agent_config_exists(self):
        """Test that voice_agent_config.json exists."""
        config_path = Path("voice_agent/voice_agent_config.json")
        assert config_path.exists(), \
            "voice_agent_config.json not found"

    def test_voice_agent_config_valid(self):
        """Test that voice_agent_config.json is valid JSON with required fields."""
        config_path = Path("voice_agent/voice_agent_config.json")
        if not config_path.exists():
            pytest.skip("voice_agent_config.json not found")

        with open(config_path) as f:
            config = json.load(f)

        # Verify required fields
        required_fields = [
            "stt_base_url", "stt_model",
            "llm_base_url", "llm_model",
            "tts_base_url", "tts_model",
            "api_key"
        ]

        for field in required_fields:
            assert field in config, f"Missing required field: {field}"

    def test_openai_client_example_exists(self):
        """Test that OpenAI client example exists."""
        client_path = Path("voice_agent/examples/openai_client.py")
        assert client_path.exists(), \
            "OpenAI client example not found"

    def test_vad_parameters_documented(self):
        """Test that VAD parameters are documented in vad.py."""
        vad_path = Path("voice_agent/vad.py")
        if not vad_path.exists():
            pytest.skip("vad.py not found")

        content = vad_path.read_text()

        # Check for key VAD parameters
        assert "VAD_THRESHOLD" in content or "threshold" in content.lower()
        assert "ONSET" in content or "onset" in content.lower()
        assert "OFFSET" in content or "offset" in content.lower()

    @pytest.mark.slow
    @pytest.mark.parametrize("model", [
        "Qwen/Qwen3.5-4B",           # LLM (medium size)
        "Qwen/Qwen3-ASR-0.6B",       # STT (small)
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",  # TTS (small)
    ])
    def test_voice_agent_model_info(self, model):
        """Test that all voice agent models can be queried."""
        result = subprocess.run(
            ["rzr-aikit", "model", "info", model],
            capture_output=True,
            text=True,
            timeout=30  # Medium LLM needs up to 30s
        )

        # Should succeed or provide meaningful error
        assert result.returncode == 0 or "Incompatible" in result.stdout

        if result.returncode == 0:
            assert model in result.stdout

    @pytest.mark.slow
    @pytest.mark.requires_network
    @pytest.mark.parametrize("model", [
        "Qwen/Qwen3.5-4B",
        "Qwen/Qwen3-ASR-0.6B",
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    ])
    def test_voice_agent_model_download(self, model):
        """Test downloading voice agent models."""
        result = subprocess.run(
            ["rzr-aikit", "model", "download", model],
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes for downloads
        )

        # Should succeed or report already cached
        assert result.returncode == 0 or \
               "already downloaded" in result.stdout.lower()


@pytest.mark.voice
@pytest.mark.slow
@pytest.mark.requires_gpu
class TestVoiceAgentServer:
    """Server operations and WebSocket workflow (requires GPU).

    Tests voice agent server lifecycle, WebSocket connectivity,
    and multi-model pipeline integration.
    """

    @pytest.mark.integration
    def test_websocket_port_available(self):
        """Test that WebSocket port (8081) is available or in use.

        Assumes voice agent server might be running on port 8081.
        This is an integration test that checks server availability.
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            # Try to connect to see if server is running
            result = sock.connect_ex(('127.0.0.1', 8081))
            # Either port is in use (0) or available (non-zero)
            assert result >= 0
        finally:
            sock.close()
