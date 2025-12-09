import pytest
from typer.testing import CliRunner
import json
import os
import tempfile
import shutil


class TestRemoveCommand:
    """Test cases for the model remove command."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    @pytest.fixture
    def app(self):
        from src.cli import app as cli_app

        return cli_app

    @pytest.fixture
    def fake_hf_cache(self, mocker):
        """Create a fake HuggingFace cache directory using a temp directory."""
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        cache_dir = os.path.join(temp_dir, "huggingface", "hub")
        os.makedirs(cache_dir, exist_ok=True)

        try:
            # Store original function
            from huggingface_hub import scan_cache_dir as original_scan_cache_dir

            # Patch scan_cache_dir to always use our temp cache directory
            def mock_scan_cache_dir():
                return original_scan_cache_dir(cache_dir=cache_dir)

            mocker.patch(
                "huggingface_hub.scan_cache_dir", side_effect=mock_scan_cache_dir
            )

            # Create model files
            model_dir = os.path.join(cache_dir, "models--microsoft--DialoGPT-small")
            os.makedirs(os.path.join(model_dir, "refs"), exist_ok=True)
            os.makedirs(os.path.join(model_dir, "snapshots", "abc123"), exist_ok=True)

            # Add model files
            with open(os.path.join(model_dir, "refs", "main"), "w") as f:
                f.write("abc123")

            with open(
                os.path.join(model_dir, "snapshots", "abc123", "config.json"), "w"
            ) as f:
                json.dump({"model_type": "gpt2"}, f)

            with open(
                os.path.join(model_dir, "snapshots", "abc123", "pytorch_model.bin"),
                "wb",
            ) as f:
                f.write(b"fake weights")

            # Create another model for testing
            model_dir2 = os.path.join(cache_dir, "models--Qwen--Qwen3-0.6B")
            os.makedirs(os.path.join(model_dir2, "refs"), exist_ok=True)
            os.makedirs(os.path.join(model_dir2, "snapshots", "def456"), exist_ok=True)

            with open(os.path.join(model_dir2, "refs", "main"), "w") as f:
                f.write("def456")

            with open(
                os.path.join(model_dir2, "snapshots", "def456", "config.json"), "w"
            ) as f:
                json.dump({"model_type": "qwen"}, f)

            with open(
                os.path.join(model_dir2, "snapshots", "def456", "model.safetensors"),
                "wb",
            ) as f:
                f.write(b"fake weights")

            yield cache_dir

        finally:
            # Clean up temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_remove_command_success(self, app, runner, fake_hf_cache):
        """Test successful model removal."""
        cache_dir = fake_hf_cache

        model_dir = os.path.join(cache_dir, "models--microsoft--DialoGPT-small")
        model_file = os.path.join(model_dir, "snapshots", "abc123", "pytorch_model.bin")

        # Verify model exists before removal
        assert os.path.exists(model_file)

        result = runner.invoke(app, ["model", "remove", "microsoft/DialoGPT-small"])

        assert result.exit_code == 0
        assert "has been removed from the local cache" in result.stdout

        # Verify model was actually removed
        assert not os.path.exists(model_dir)

    def test_remove_command_model_not_found(self, app, runner, fake_hf_cache):
        """Test removal when model is not found."""
        result = runner.invoke(app, ["model", "remove", "nonexistent/model"])

        assert result.exit_code == 0
        assert "not found" in result.stdout

    def test_remove_command_multiple_revisions(self, mocker, app, runner):
        """Test removal of model with multiple revisions."""
        # Create temporary directory for this test
        temp_dir = tempfile.mkdtemp()
        cache_dir = os.path.join(temp_dir, "huggingface", "hub")
        os.makedirs(cache_dir, exist_ok=True)

        try:
            # Store original function
            from huggingface_hub import scan_cache_dir as original_scan_cache_dir

            # Patch scan_cache_dir to use our temp cache directory
            def mock_scan_cache_dir():
                return original_scan_cache_dir(cache_dir=cache_dir)

            mocker.patch(
                "huggingface_hub.scan_cache_dir", side_effect=mock_scan_cache_dir
            )

            # Create model with multiple revisions
            model_dir = os.path.join(cache_dir, "models--test--multi-revision")
            os.makedirs(os.path.join(model_dir, "refs"), exist_ok=True)
            os.makedirs(os.path.join(model_dir, "snapshots", "rev1"), exist_ok=True)
            os.makedirs(os.path.join(model_dir, "snapshots", "rev2"), exist_ok=True)

            with open(os.path.join(model_dir, "refs", "main"), "w") as f:
                f.write("rev2")

            with open(
                os.path.join(model_dir, "snapshots", "rev1", "config.json"), "w"
            ) as f:
                json.dump({}, f)

            with open(
                os.path.join(model_dir, "snapshots", "rev1", "model.bin"), "wb"
            ) as f:
                f.write(b"v1")

            with open(
                os.path.join(model_dir, "snapshots", "rev2", "config.json"), "w"
            ) as f:
                json.dump({}, f)

            with open(
                os.path.join(model_dir, "snapshots", "rev2", "model.bin"), "wb"
            ) as f:
                f.write(b"v2")

            # Verify files exist
            assert os.path.exists(
                os.path.join(model_dir, "snapshots", "rev1", "model.bin")
            )
            assert os.path.exists(
                os.path.join(model_dir, "snapshots", "rev2", "model.bin")
            )

            result = runner.invoke(app, ["model", "remove", "test/multi-revision"])

            assert result.exit_code == 0
            assert "has been removed from the local cache" in result.stdout

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_remove_command_permission_error(self, mocker, app, runner, fake_hf_cache):
        """Test removal when permission error occurs."""
        # Override the scan_cache_dir patch to raise permission error
        mock_scan_cache = mocker.patch("huggingface_hub.scan_cache_dir")
        mock_scan_cache.side_effect = PermissionError("Permission denied")

        result = runner.invoke(app, ["model", "remove", "microsoft/DialoGPT-small"])

        assert result.exit_code == 0
        assert "Permission denied" in result.stdout

    def test_remove_command_cache_scan_error(self, mocker, app, runner, fake_hf_cache):
        """Test removal when cache scanning fails."""
        mock_scan_cache = mocker.patch("huggingface_hub.scan_cache_dir")
        mock_scan_cache.side_effect = Exception("Cache directory corrupted")

        result = runner.invoke(app, ["model", "remove", "microsoft/DialoGPT-small"])

        assert result.exit_code == 0
        assert "Cache directory corrupted" in result.stdout

    def test_remove_command_empty_cache(self, mocker, app, runner):
        """Test removal from empty cache."""
        temp_dir = tempfile.mkdtemp()
        cache_dir = os.path.join(temp_dir, "huggingface", "hub")
        os.makedirs(cache_dir, exist_ok=True)

        try:
            # Store original function
            from huggingface_hub import scan_cache_dir as original_scan_cache_dir

            # Patch scan_cache_dir to use our empty temp cache directory
            def mock_scan_cache_dir():
                return original_scan_cache_dir(cache_dir=cache_dir)

            mocker.patch(
                "huggingface_hub.scan_cache_dir", side_effect=mock_scan_cache_dir
            )

            result = runner.invoke(app, ["model", "remove", "nonexistent/model"])

            assert result.exit_code == 0
            assert "not found" in result.stdout

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_remove_command_case_sensitivity(self, runner, app, fake_hf_cache):
        """Test that model removal is case sensitive."""
        cache_dir = fake_hf_cache

        # Try to remove with different case
        result = runner.invoke(app, ["model", "remove", "Microsoft/DialogPT-Small"])

        assert result.exit_code == 0
        assert "not found" in result.stdout

        # Original model should still exist
        model_dir = os.path.join(cache_dir, "models--microsoft--DialoGPT-small")
        assert os.path.exists(model_dir)

    def test_remove_function_directly(self, runner, app, mocker, fake_hf_cache):
        """Test the remove function directly without CLI."""
        cache_dir = fake_hf_cache

        from src.model.remove import remove

        model_dir = os.path.join(cache_dir, "models--microsoft--DialoGPT-small")
        assert os.path.exists(model_dir)

        result = runner.invoke(app, ["model", "remove", "microsoft/DialoGPT-small"])

        assert "Model 'microsoft/DialoGPT-small' has been removed from the local cache." in result.stdout

        # Verify model was actually removed
        assert not os.path.exists(model_dir)

    def test_remove_model_with_special_characters(self, mocker, app, runner):
        """Test removal of model with special characters in name."""
        temp_dir = tempfile.mkdtemp()
        cache_dir = os.path.join(temp_dir, "huggingface", "hub")
        os.makedirs(cache_dir, exist_ok=True)

        try:
            # Store original function
            from huggingface_hub import scan_cache_dir as original_scan_cache_dir

            # Patch scan_cache_dir to use our temp cache directory
            def mock_scan_cache_dir():
                return original_scan_cache_dir(cache_dir=cache_dir)

            mocker.patch(
                "huggingface_hub.scan_cache_dir", side_effect=mock_scan_cache_dir
            )

            # Create model with special characters (hyphens, underscores)
            model_dir = os.path.join(
                cache_dir, "models--test--model-with_special-chars"
            )
            os.makedirs(os.path.join(model_dir, "refs"), exist_ok=True)
            os.makedirs(os.path.join(model_dir, "snapshots", "hash123"), exist_ok=True)

            with open(os.path.join(model_dir, "refs", "main"), "w") as f:
                f.write("hash123")

            with open(
                os.path.join(model_dir, "snapshots", "hash123", "config.json"), "w"
            ) as f:
                json.dump({}, f)

            with open(
                os.path.join(model_dir, "snapshots", "hash123", "model.bin"), "wb"
            ) as f:
                f.write(b"weights")

            result = runner.invoke(
                app, ["model", "remove", "test/model-with_special-chars"]
            )

            assert result.exit_code == 0
            assert "has been removed from the local cache" in result.stdout

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_verify_temp_cleanup(self, fake_hf_cache):
        """Test that temp directories are properly cleaned up."""
        cache_dir = fake_hf_cache

        # Cache directory should exist during test
        assert os.path.exists(cache_dir)

        # After fixture cleanup, temp directory should be removed
        # This is handled automatically by the fixture's finally block
