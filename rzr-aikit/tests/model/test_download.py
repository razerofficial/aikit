import os
import pytest
from pytest_mock import mocker
from typer.testing import CliRunner

class TestDownloadCommand:
    """Test cases for the model download command."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    @pytest.fixture
    def app(self):
        from src.cli import app as cli_app

        return cli_app

    def test_download_command_new_model(self, mocker, app, runner):
        """Test downloading a model that's not cached."""
        mock_check_downloaded = mocker.patch("src.model.download.check_if_downloaded")
        mock_download_model = mocker.patch("src.model.download.download_model")

        mock_check_downloaded.return_value = False

        result = runner.invoke(app, ["model", "download", "Qwen/Qwen3-0.6B"])

        assert result.exit_code == 0
        mock_check_downloaded.assert_called_once_with("Qwen/Qwen3-0.6B")
        mock_download_model.assert_called_once_with("Qwen/Qwen3-0.6B")

    def test_download_command_already_cached(self, mocker, app, runner):
        """Test downloading a model that's already cached."""
        mock_check_downloaded = mocker.patch("src.model.download.check_if_downloaded")
        mock_download_model = mocker.patch("src.model.download.download_model")

        mock_check_downloaded.return_value = True

        result = runner.invoke(app, ["model", "download", "Qwen/Qwen3-0.6B"])

        assert result.exit_code == 0
        assert "already downloaded" in result.stdout
        mock_check_downloaded.assert_called_once_with("Qwen/Qwen3-0.6B")
        mock_download_model.assert_not_called()

    def test_download_command_default_model(self, mocker, app, runner):
        """Test download command with default model."""
        mock_check_downloaded = mocker.patch("src.model.download.check_if_downloaded")
        mock_download_model = mocker.patch("src.model.download.download_model")

        mock_check_downloaded.return_value = False

        result = runner.invoke(app, ["model", "download"])

        assert result.exit_code == 0
        mock_check_downloaded.assert_called_once_with("facebook/opt-125m")
        mock_download_model.assert_called_once_with("facebook/opt-125m")

    def test_download_command_default_model_already_cached(self, mocker, app, runner):
        """Test download command with default model already cached."""
        mock_check_downloaded = mocker.patch("src.model.download.check_if_downloaded")
        mock_download_model = mocker.patch("src.model.download.download_model")

        mock_check_downloaded.return_value = True

        result = runner.invoke(app, ["model", "download"])

        assert result.exit_code == 0
        assert "already downloaded" in result.stdout
        mock_check_downloaded.assert_called_once_with("facebook/opt-125m")
        mock_download_model.assert_not_called()

    def test_check_if_downloaded_true(self, mocker):
        """Test check_if_downloaded when model is cached with weights."""
        mock_scan_cache = mocker.patch("huggingface_hub.scan_cache_dir")
        mock_has_weights = mocker.patch("src.model.list.has_weights")

        from src.model.download import check_if_downloaded

        repo = mocker.Mock()
        repo.repo_id = "Qwen/Qwen3-0.6B"

        cache_info = mocker.Mock()
        cache_info.repos = [repo]
        mock_scan_cache.return_value = cache_info
        mock_has_weights.return_value = True

        result = check_if_downloaded("Qwen/Qwen3-0.6B")

        assert result == True
        mock_has_weights.assert_called_once_with(repo)

    def test_check_if_downloaded_no_weights(self, mocker):
        """Test check_if_downloaded when model is cached but without weights."""
        mock_scan_cache = mocker.patch("huggingface_hub.scan_cache_dir")
        mock_has_weights = mocker.patch("src.model.list.has_weights")

        from src.model.download import check_if_downloaded

        repo = mocker.Mock()
        repo.repo_id = "Qwen/Qwen3-0.6B"

        cache_info = mocker.Mock()
        cache_info.repos = [repo]
        mock_scan_cache.return_value = cache_info
        mock_has_weights.return_value = False

        result = check_if_downloaded("Qwen/Qwen3-0.6B")

        assert result == False

    def test_check_if_downloaded_not_in_cache(self, mocker):
        """Test check_if_downloaded when model is not in cache."""
        mock_scan_cache = mocker.patch("huggingface_hub.scan_cache_dir")

        from src.model.download import check_if_downloaded

        cache_info = mocker.Mock()
        cache_info.repos = []
        mock_scan_cache.return_value = cache_info

        result = check_if_downloaded("Qwen/Qwen3-0.6B")

        assert result == False

    def test_check_if_downloaded_different_model(self, mocker):
        """Test check_if_downloaded when different model is in cache."""
        mock_scan_cache = mocker.patch("huggingface_hub.scan_cache_dir")

        from src.model.download import check_if_downloaded

        repo = mocker.Mock()
        repo.repo_id = "different/model"

        cache_info = mocker.Mock()
        cache_info.repos = [repo]
        mock_scan_cache.return_value = cache_info

        result = check_if_downloaded("Qwen/Qwen3-0.6B")

        assert result == False

    def test_check_if_downloaded_exception_handling(self, mocker):
        """Test check_if_downloaded when cache scanning raises an exception."""
        mock_scan_cache = mocker.patch("huggingface_hub.scan_cache_dir")

        from src.model.download import check_if_downloaded

        mock_scan_cache.side_effect = Exception("Cache directory not accessible")

        result = check_if_downloaded("Qwen/Qwen3-0.6B")

        assert result == False

    def test_download_model_success(self, mocker, app, runner):
        """Test successful model download through CLI."""
        # Mock at function level - don't test implementation details
        mock_check_downloaded = mocker.patch("src.model.download.check_if_downloaded")
        mock_download_model = mocker.patch("src.model.download.download_model")

        mock_check_downloaded.return_value = False

        result = runner.invoke(app, ["model", "download", "Qwen/Qwen3-0.6B"])

        assert result.exit_code == 0
        mock_check_downloaded.assert_called_once_with("Qwen/Qwen3-0.6B")
        mock_download_model.assert_called_once_with("Qwen/Qwen3-0.6B")

    def test_download_model_failure_with_cleanup(self, mocker, app, runner):
        """Test model download failure through CLI."""
        mock_check_downloaded = mocker.patch("src.model.download.check_if_downloaded")
        mock_download_model = mocker.patch("src.model.download.download_model")

        mock_check_downloaded.return_value = False
        mock_download_model.side_effect = Exception("Download failed")

        result = runner.invoke(app, ["model", "download", "Qwen/Qwen3-0.6B"])

        # CLI should handle the exception gracefully
        assert result.exit_code == 1
        mock_check_downloaded.assert_called_once_with("Qwen/Qwen3-0.6B")
        mock_download_model.assert_called_once_with("Qwen/Qwen3-0.6B")

    def test_download_command_network_error_during_check(self, mocker, app, runner):
        """Test download command when network error occurs during check."""
        mock_check_downloaded = mocker.patch("src.model.download.check_if_downloaded")
        mock_download_model = mocker.patch("src.model.download.download_model")

        mock_check_downloaded.side_effect = Exception("Network error")

        # Should still attempt download even if check fails
        result = runner.invoke(app, ["model", "download", "Qwen/Qwen3-0.6B"])

        assert result.exit_code == 1

    def test_download_command_permission_error(self, mocker, app, runner):
        """Test download command with permission error during download."""
        mock_check_downloaded = mocker.patch("src.model.download.check_if_downloaded")
        mock_download_model = mocker.patch("src.model.download.download_model")

        mock_check_downloaded.return_value = False
        mock_download_model.side_effect = PermissionError("Permission denied")

        result = runner.invoke(app, ["model", "download", "Qwen/Qwen3-0.6B"])

        assert result.exit_code == 1

    def test_check_if_downloaded_multiple_repos(self, mocker):
        """Test check_if_downloaded with multiple repositories in cache."""
        mock_scan_cache = mocker.patch("huggingface_hub.scan_cache_dir")
        mock_has_weights = mocker.patch("src.model.list.has_weights")

        from src.model.download import check_if_downloaded

        repo1 = mocker.Mock()
        repo1.repo_id = "model/one"
        repo2 = mocker.Mock()
        repo2.repo_id = "Qwen/Qwen3-0.6B"
        repo3 = mocker.Mock()
        repo3.repo_id = "model/three"

        cache_info = mocker.Mock()
        cache_info.repos = [repo1, repo2, repo3]
        mock_scan_cache.return_value = cache_info
        mock_has_weights.return_value = True

        result = check_if_downloaded("Qwen/Qwen3-0.6B")

        assert result == True
        mock_has_weights.assert_called_once_with(repo2)
