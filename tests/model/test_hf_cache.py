import os
import pytest
from huggingface_hub.errors import LocalEntryNotFoundError


class TestResolveCachedPath:

    def test_local_directory_is_returned_directly(self, tmp_path):
        """An existing local directory path is returned as-is without hitting the cache."""
        from rzr_aikit.utils.hf_cache import resolve_cached_path

        result = resolve_cached_path(str(tmp_path))

        assert result == str(tmp_path)

    def test_cached_model_returns_snapshot_path(self, mocker):
        """A model present in the HF cache returns its snapshot directory."""
        mock_snapshot = mocker.patch(
            "rzr_aikit.utils.hf_cache.snapshot_download",
            return_value="/cache/models--org--model/snapshots/abc123",
        )

        from rzr_aikit.utils.hf_cache import resolve_cached_path
        result = resolve_cached_path("org/model")

        assert result == "/cache/models--org--model/snapshots/abc123"
        mock_snapshot.assert_called_once_with(
            repo_id="org/model",
            local_files_only=True,
            ignore_patterns=None,
        )

    def test_missing_model_returns_none(self, mocker):
        """LocalEntryNotFoundError from snapshot_download yields None."""
        mocker.patch(
            "rzr_aikit.utils.hf_cache.snapshot_download",
            side_effect=LocalEntryNotFoundError("not cached"),
        )

        from rzr_aikit.utils.hf_cache import resolve_cached_path
        assert resolve_cached_path("org/model") is None

    def test_unexpected_exception_returns_none(self, mocker):
        """Any other exception from snapshot_download yields None."""
        mocker.patch(
            "rzr_aikit.utils.hf_cache.snapshot_download",
            side_effect=OSError("disk read error"),
        )

        from rzr_aikit.utils.hf_cache import resolve_cached_path
        assert resolve_cached_path("org/model") is None

    def test_nonexistent_string_path_falls_through_to_cache(self, mocker):
        """A string that is not a directory is looked up in the HF cache."""
        mock_snapshot = mocker.patch(
            "rzr_aikit.utils.hf_cache.snapshot_download",
            side_effect=LocalEntryNotFoundError("not cached"),
        )

        from rzr_aikit.utils.hf_cache import resolve_cached_path
        result = resolve_cached_path("not/a/local/path")

        assert result is None
        mock_snapshot.assert_called_once()


class TestIsCached:

    def test_returns_true_when_model_is_cached(self, mocker):
        mocker.patch(
            "rzr_aikit.utils.hf_cache.resolve_cached_path",
            return_value="/some/path",
        )

        from rzr_aikit.utils.hf_cache import is_cached
        assert is_cached("org/model") is True

    def test_returns_false_when_model_is_not_cached(self, mocker):
        mocker.patch(
            "rzr_aikit.utils.hf_cache.resolve_cached_path",
            return_value=None,
        )

        from rzr_aikit.utils.hf_cache import is_cached
        assert is_cached("org/model") is False

    def test_local_directory_is_cached(self, tmp_path):
        from rzr_aikit.utils.hf_cache import is_cached
        assert is_cached(str(tmp_path)) is True
