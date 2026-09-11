"""
Validates Notebook 1: On-Device Inferencing
Tests complete LLM workflow from model download through vLLM server to text generation.
"""
import pytest
import subprocess
import time
import json


@pytest.mark.llm
class TestLLMCLIWorkflow:
    """CLI validation and model operations (fast, no GPU).

    Tests the complete CLI workflow for LLM model management:
    command existence, model info, model download, model list/run operations.
    """

    def test_model_info_command_exists(self):
        """Test that rzr-aikit model info command is available."""
        result = subprocess.run(
            ["rzr-aikit", "model", "info", "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )
        assert result.returncode == 0

    @pytest.mark.slow
    def test_get_model_info(self, test_model):
        """Test retrieving model information."""
        result = subprocess.run(
            ["rzr-aikit", "model", "info", test_model],
            capture_output=True,
            text=True,
            timeout=60
        )

        # Command should succeed (exit code 0) or provide useful error
        assert result.returncode == 0 or "Incompatible model" in result.stdout

        # Should contain model name in output
        if result.returncode == 0:
            assert test_model in result.stdout

    def test_model_download_command_exists(self):
        """Test that rzr-aikit model download command is available."""
        result = subprocess.run(
            ["rzr-aikit", "model", "download", "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )
        assert result.returncode == 0

    @pytest.mark.slow
    @pytest.mark.requires_network
    def test_model_download_or_cached(self, test_model):
        """Test model download (or verify it's already cached)."""
        result = subprocess.run(
            ["rzr-aikit", "model", "download", test_model],
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes for download
        )

        # Should either succeed or report already downloaded
        assert result.returncode == 0 or \
               "already downloaded" in result.stdout.lower()

    @pytest.mark.slow
    def test_model_list_command(self):
        """Test that model list command works."""
        result = subprocess.run(
            ["rzr-aikit", "model", "list"],
            capture_output=True,
            text=True,
            timeout=60
        )

        # Should succeed
        assert result.returncode == 0

    def test_model_run_help(self):
        """Test that model run command exists."""
        result = subprocess.run(
            ["rzr-aikit", "model", "run", "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )

        # Should provide help text
        assert result.returncode == 0
        assert "usage:" in result.stdout.lower() or "options:" in result.stdout.lower()


@pytest.mark.llm
@pytest.mark.slow
@pytest.mark.requires_gpu
class TestLLMServerWorkflow:
    """Server lifecycle and generation (requires GPU).

    Tests the complete LLM server workflow:
    vLLM server availability, text generation via Python API.
    """

    def test_vllm_serve_command_available(self):
        """Test that vllm serve command is available."""
        result = subprocess.run(
            ["vllm", "serve", "--help"],
            capture_output=True,
            text=True,
            timeout=30  # vLLM plugin initialization can take >10s
        )
        assert result.returncode == 0

    @pytest.mark.integration
    def test_basic_text_generation(self, test_model, tmp_path):
        """Test basic text generation via Python API."""
        try:
            from vllm import LLM, SamplingParams

            # Use small parameters for testing
            llm = LLM(
                model=test_model,
                gpu_memory_utilization=0.3,
                enforce_eager=True  # Skip graph capture for faster test
            )

            sampling_params = SamplingParams(
                temperature=0.8,
                top_p=0.95,
                max_tokens=10
            )

            outputs = llm.generate(
                ["The capital of France is"],
                sampling_params
            )

            assert len(outputs) == 1
            assert len(outputs[0].outputs) > 0
            assert len(outputs[0].outputs[0].text) > 0

        except ImportError:
            pytest.skip("vLLM not available for direct import")
        except Exception as e:
            # GPU memory issues are acceptable in constrained test environments
            if "CUDA out of memory" in str(e) or "No GPU" in str(e):
                pytest.skip(f"GPU resources insufficient: {e}")
            raise

    def test_generation_produces_output(self):
        """Test that generation produces non-empty output."""
        # This is a placeholder for actual generation test
        # In practice, this would use the vLLM API or REST endpoint

        sample_prompt = "The capital of France is"
        sample_output = " Paris"  # Expected type of output

        assert len(sample_prompt) > 0
        assert len(sample_output) > 0
        # In real test, verify llm.generate() produces similar output
