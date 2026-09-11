import pytest
from typer.testing import CliRunner


class TestDownloadCommand:
    """CLI tests for the model download command."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    @pytest.fixture
    def app(self):
        from rzr_aikit.cli import app as cli_app
        return cli_app

    def test_download_new_model(self, mocker, app, runner):
        """Not-cached model triggers download_model."""
        mocker.patch("rzr_aikit.utils.hf_cache.is_cached", return_value=False)
        mock_download = mocker.patch("rzr_aikit.model.download.download_model")

        result = runner.invoke(app, ["model", "download", "Qwen/Qwen3-0.6B"])

        assert result.exit_code == 0
        mock_download.assert_called_once_with("Qwen/Qwen3-0.6B")

    def test_download_already_cached(self, mocker, app, runner):
        """Already-cached model skips download and prints a message."""
        mocker.patch("rzr_aikit.utils.hf_cache.is_cached", return_value=True)
        mock_download = mocker.patch("rzr_aikit.model.download.download_model")

        result = runner.invoke(app, ["model", "download", "Qwen/Qwen3-0.6B"])

        assert result.exit_code == 0
        assert "already downloaded" in result.stdout
        mock_download.assert_not_called()

    def test_download_default_model(self, mocker, app, runner):
        """Default model is facebook/opt-125m."""
        mocker.patch("rzr_aikit.utils.hf_cache.is_cached", return_value=False)
        mock_download = mocker.patch("rzr_aikit.model.download.download_model")

        result = runner.invoke(app, ["model", "download"])

        assert result.exit_code == 0
        mock_download.assert_called_once_with("facebook/opt-125m")

    def test_download_default_model_already_cached(self, mocker, app, runner):
        """Default model already cached skips download."""
        mocker.patch("rzr_aikit.utils.hf_cache.is_cached", return_value=True)
        mock_download = mocker.patch("rzr_aikit.model.download.download_model")

        result = runner.invoke(app, ["model", "download"])

        assert result.exit_code == 0
        assert "already downloaded" in result.stdout
        mock_download.assert_not_called()


class TestDownloadModel:
    """Unit tests for the download_model helper."""

    def test_success_calls_snapshot_download(self, mocker):
        """snapshot_download is called with local_files_only=False."""
        mock_snapshot = mocker.patch("huggingface_hub.snapshot_download")

        from rzr_aikit.model.download import download_model
        download_model("org/model")

        mock_snapshot.assert_called_once_with(repo_id="org/model", local_files_only=False)

    def test_failure_triggers_remove(self, mocker):
        """When snapshot_download raises, remove is called to clean up the partial cache."""
        mocker.patch(
            "huggingface_hub.snapshot_download",
            side_effect=Exception("network error"),
        )
        mock_remove = mocker.patch("rzr_aikit.model.remove.remove")

        from rzr_aikit.model.download import download_model
        download_model("org/model")

        mock_remove.assert_called_once_with("org/model")

    def test_failure_remove_exception_is_swallowed(self, mocker):
        """An exception from remove does not propagate to the caller."""
        mocker.patch(
            "huggingface_hub.snapshot_download",
            side_effect=Exception("network error"),
        )
        mocker.patch(
            "rzr_aikit.model.remove.remove",
            side_effect=PermissionError("no access"),
        )

        from rzr_aikit.model.download import download_model
        download_model("org/model")  # must not raise
