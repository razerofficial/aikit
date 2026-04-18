import pytest
from typer.testing import CliRunner


class TestInfoCommand:
    """Test cases for the model info command."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    @pytest.fixture
    def app(self):
        from rzr_aikit.cli import app as cli_app

        return cli_app

    @pytest.fixture
    def mock_fetcher_hub(self, mocker):
        """Mock ModelInfoFetcher for a hub model (Qwen/Qwen3-0.6B)."""
        fetcher = mocker.Mock()
        fetcher.source = "hub"
        fetcher.total_bytes = 600_000_000  # 600 MB
        fetcher.dtype = "float16"
        fetcher.config = {
            "max_position_embeddings": 4096,
            "hidden_size": 768,
            "num_attention_heads": 12,
        }
        fetcher.quant_config = {}

        model_info = mocker.Mock()
        model_info.id = "Qwen/Qwen3-0.6B"
        model_info.pipeline_tag = "text-generation"
        model_info.downloads = 12345
        model_info.last_modified = "2024-01-01T00:00:00Z"
        card_data = mocker.Mock()
        card_data.license = "apache-2.0"
        model_info.card_data = card_data
        fetcher._model_info = model_info  # pylint: disable=protected-access

        return fetcher

    @pytest.fixture
    def mock_fetcher_local(self, mocker):
        """Mock ModelInfoFetcher for a local/cache model."""
        fetcher = mocker.Mock()
        fetcher.source = "local"
        fetcher.total_bytes = 600_000_000
        fetcher.dtype = "float16"
        fetcher.config = {
            "max_position_embeddings": 4096,
            "hidden_size": 768,
            "num_attention_heads": 12,
        }
        fetcher.quant_config = {}
        return fetcher

    def _patch_common(self, mocker, *, connected=True):
        """Patch mocks shared by most tests. Returns (fetcher_class, get_vram, diffusion_class)."""
        mock_fetcher_class = mocker.patch(
            "rzr_aikit.utils.ModelInfoFetcher.ModelInfoFetcher"
        )
        mock_get_vram = mocker.patch("rzr_aikit.utils.mlib.get_cuda_total_vram")
        mock_diffusion = mocker.patch(
            "rzr_aikit.utils.DiffusionModelInfoFetcher.DiffusionModelInfoFetcher"
        )
        mocker.patch(
            "rzr_aikit.utils.connectivity.check_huggingface_connectivity",
            return_value=connected,
        )
        return mock_fetcher_class, mock_get_vram, mock_diffusion

    # ------------------------------------------------------------------
    # Hub model tests
    # ------------------------------------------------------------------

    def test_info_command_hub_model_nongated(
        self, mocker, app, runner, mock_fetcher_hub
    ):
        """Non-gated hub model shows type, downloads, dtype and context length."""
        mock_fetcher_class, mock_get_vram, _ = self._patch_common(mocker)
        mock_fetcher_class.return_value = mock_fetcher_hub
        mock_get_vram.side_effect = [48 * 1024**3, 96 * 1024**3]

        result = runner.invoke(app, ["model", "info", "Qwen/Qwen3-0.6B"])

        assert result.exit_code == 0
        assert "✅ Yes" in result.stdout
        assert "Qwen/Qwen3-0.6B" in result.stdout
        assert "text-generation" in result.stdout
        assert "float16" in result.stdout
        assert "4096" in result.stdout
        assert "12,345" in result.stdout  # formatted downloads

    def test_info_command_hub_model_gated_with_auth(self, mocker, app, runner):
        """Gated hub model succeeds when a valid token is supplied."""
        mock_fetcher_class, mock_get_vram, _ = self._patch_common(mocker)
        from rzr_aikit import HuggingfaceAccessTokenRequired

        gated_fetcher = mocker.Mock()
        gated_fetcher.source = "hub"
        gated_fetcher.total_bytes = 1_200_000_000
        gated_fetcher.dtype = "bfloat16"
        gated_fetcher.config = {"max_position_embeddings": 8192, "hidden_size": 2048}
        gated_fetcher.quant_config = {}

        model_info = mocker.Mock()
        model_info.id = "google/gemma-3-1b-it"
        model_info.pipeline_tag = "text-generation"
        model_info.downloads = 50000
        model_info.last_modified = "2024-02-01T00:00:00Z"
        card_data = mocker.Mock()
        card_data.license = "gemma"
        model_info.card_data = card_data
        gated_fetcher._model_info = model_info  # pylint: disable=protected-access

        mock_fetcher_class.side_effect = [
            HuggingfaceAccessTokenRequired("Token required"),
            gated_fetcher,
        ]

        mock_token_class = mocker.patch(
            "rzr_aikit.utils.HuggingfaceHubToken.HuggingfaceHubToken"
        )
        mock_token_instance = mocker.Mock()
        mock_token_instance.get_access_token.return_value = "hf_test_token_123"
        mock_token_class.return_value = mock_token_instance

        mock_get_vram.side_effect = [24 * 1024**3, 0]

        result = runner.invoke(app, ["model", "info", "google/gemma-3-1b-it"])

        assert result.exit_code == 0
        assert "✅ Yes" in result.stdout
        assert "google/gemma-3-1b-it" in result.stdout
        assert "bfloat16" in result.stdout
        assert mock_token_instance.get_access_token.called

    def test_info_command_hub_model_gated_without_auth(self, mocker, app, runner):
        """Gated hub model without a token fails gracefully."""
        mock_fetcher_class, _, _ = self._patch_common(mocker)
        from rzr_aikit import HuggingfaceAccessTokenRequired

        mock_fetcher_class.side_effect = HuggingfaceAccessTokenRequired("Token required")

        mock_token_class = mocker.patch(
            "rzr_aikit.utils.HuggingfaceHubToken.HuggingfaceHubToken"
        )
        mock_token_instance = mocker.Mock()
        mock_token_instance.get_access_token.side_effect = HuggingfaceAccessTokenRequired(
            "No token available"
        )
        mock_token_class.return_value = mock_token_instance

        result = runner.invoke(app, ["model", "info", "google/gemma-3-1b-it"])

        assert result.exit_code != 0 or "Incompatible model" in result.stdout

    # ------------------------------------------------------------------
    # Local / cache model tests
    # ------------------------------------------------------------------

    def test_info_command_local_model(self, mocker, app, runner):
        """Local model with HF connectivity shows hub info alongside local data."""
        mock_fetcher_class, mock_get_vram, _ = self._patch_common(mocker)

        local_fetcher = mocker.Mock()
        local_fetcher.source = "local"
        local_fetcher.total_bytes = 600_000_000
        local_fetcher.dtype = "float16"
        local_fetcher.config = {
            "max_position_embeddings": 4096,
            "hidden_size": 768,
            "num_attention_heads": 12,
        }
        local_fetcher.quant_config = {}

        model_info = mocker.Mock()
        model_info.id = "local/model"
        model_info.pipeline_tag = "text-generation"
        model_info.downloads = 5000
        model_info.last_modified = "2024-01-15T00:00:00Z"
        card_data = mocker.Mock()
        card_data.license = "mit"
        model_info.card_data = card_data
        local_fetcher._model_info = model_info  # pylint: disable=protected-access

        mock_fetcher_class.return_value = local_fetcher
        mock_get_vram.side_effect = [48 * 1024**3, 0]

        result = runner.invoke(app, ["model", "info", "/path/to/local/model"])

        assert result.exit_code == 0
        assert "✅ Yes" in result.stdout
        assert "text-generation" in result.stdout
        assert "float16" in result.stdout
        assert "4096" in result.stdout

    def test_info_command_cached_model(self, mocker, app, runner):
        """Cached model (offline) does not show Downloads."""
        mock_fetcher_class, mock_get_vram, _ = self._patch_common(mocker, connected=False)

        cached_fetcher = mocker.Mock()
        cached_fetcher.source = "cache"
        cached_fetcher.total_bytes = 600_000_000
        cached_fetcher.dtype = "float16"
        cached_fetcher.config = {"max_position_embeddings": 4096}
        cached_fetcher.quant_config = {}

        mock_fetcher_class.return_value = cached_fetcher
        mock_get_vram.side_effect = [24 * 1024**3, 0]

        result = runner.invoke(app, ["model", "info", "org/model"])

        assert result.exit_code == 0
        assert "✅ Yes" in result.stdout
        assert "Downloads:" not in result.stdout

    # ------------------------------------------------------------------
    # Compatibility tests
    # ------------------------------------------------------------------

    def test_info_command_incompatible_hardware(self, mocker, app, runner):
        """Large model on a small GPU shows ❌ No."""
        mock_fetcher_class, mock_get_vram, _ = self._patch_common(mocker)

        large_fetcher = mocker.Mock()
        large_fetcher.source = "hub"
        large_fetcher.total_bytes = 30 * 1024**3  # 30 GB
        large_fetcher.dtype = "float32"
        large_fetcher.config = {"max_position_embeddings": 8192}
        large_fetcher.quant_config = {}

        model_info = mocker.Mock()
        model_info.pipeline_tag = "text-generation"
        model_info.downloads = 1000
        model_info.last_modified = "2024-01-01T00:00:00Z"
        card_data = mocker.Mock()
        card_data.license = "apache-2.0"
        model_info.card_data = card_data
        large_fetcher._model_info = model_info  # pylint: disable=protected-access

        mock_fetcher_class.return_value = large_fetcher
        mock_get_vram.side_effect = [4 * 1024**3, 0]

        result = runner.invoke(app, ["model", "info", "large/model"])

        assert result.exit_code == 0
        assert "❌ No" in result.stdout

    def test_info_command_limited_compatibility(self, mocker, app, runner):
        """Model that fits only with 4-bit quantization shows 🟡 Limited."""
        mock_fetcher_class, mock_get_vram, _ = self._patch_common(mocker)

        model_fetcher = mocker.Mock()
        model_fetcher.source = "hub"
        model_fetcher.total_bytes = 28 * 1024**3
        model_fetcher.dtype = "float16"
        model_fetcher.config = {"max_position_embeddings": 4096}
        model_fetcher.quant_config = {}

        model_info = mocker.Mock()
        model_info.pipeline_tag = "text-generation"
        model_info.downloads = 1000
        model_info.last_modified = "2024-01-01T00:00:00Z"
        card_data = mocker.Mock()
        card_data.license = "apache-2.0"
        model_info.card_data = card_data
        model_fetcher._model_info = model_info  # pylint: disable=protected-access

        mock_fetcher_class.return_value = model_fetcher
        mock_get_vram.side_effect = [16 * 1024**3, 32 * 1024**3]

        result = runner.invoke(app, ["model", "info", "test/model"])

        assert result.exit_code == 0
        assert "🟡 Limited" in result.stdout
        assert "4-bit on-the-fly quantization" in result.stdout

    def test_info_command_no_gpu(self, mocker, app, runner, mock_fetcher_hub):
        """No GPU available → ❌ No for local compatibility."""
        mock_fetcher_class, mock_get_vram, _ = self._patch_common(mocker)
        mock_fetcher_class.return_value = mock_fetcher_hub
        mock_get_vram.side_effect = [0, 0]

        result = runner.invoke(app, ["model", "info", "test/model"])

        assert result.exit_code == 0
        assert "❌ No" in result.stdout

    def test_info_command_no_ray_cluster(self, mocker, app, runner, mock_fetcher_hub):
        """No Ray cluster → distributed column shows 'Not connected to Ray cluster'."""
        mock_fetcher_class, mock_get_vram, _ = self._patch_common(mocker)
        mock_fetcher_class.return_value = mock_fetcher_hub
        mock_get_vram.side_effect = [24 * 1024**3, 0]

        result = runner.invoke(app, ["model", "info", "test/model"])

        assert result.exit_code == 0
        assert "Not connected to Ray cluster" in result.stdout

    def test_info_command_ray_discovery_in_progress(
        self, mocker, app, runner, mock_fetcher_hub
    ):
        """GPU discovery in progress → appropriate message shown."""
        mock_fetcher_class, mock_get_vram, _ = self._patch_common(mocker)
        mock_fetcher_class.return_value = mock_fetcher_hub
        mock_get_vram.side_effect = [24 * 1024**3, -1]

        result = runner.invoke(app, ["model", "info", "test/model"])

        assert result.exit_code == 0
        assert "GPU discovery is in progress" in result.stdout

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def test_info_command_general_error(self, mocker, app, runner):
        """General exception from ModelInfoFetcher shows 'Incompatible model'."""
        mock_fetcher_class, _, _ = self._patch_common(mocker)
        mock_fetcher_class.side_effect = Exception("General model error")

        result = runner.invoke(app, ["model", "info", "test/model"])

        assert result.exit_code == 0
        assert "Incompatible model" in result.stdout

    def test_info_command_license_access_error(self, mocker, app, runner):
        """Missing card_data.license handled gracefully (no crash)."""
        mock_fetcher_class, mock_get_vram, _ = self._patch_common(mocker)

        fetcher = mocker.Mock()
        fetcher.source = "hub"
        fetcher.total_bytes = 1024**3
        fetcher.dtype = "float16"
        fetcher.config = {"max_position_embeddings": 4096}
        fetcher.quant_config = {}

        info_without_license = mocker.Mock()
        info_without_license.id = "test/model"
        info_without_license.pipeline_tag = "text-generation"
        info_without_license.downloads = 100
        info_without_license.last_modified = "2024-01-01T00:00:00Z"
        info_without_license.card_data = None
        fetcher._model_info = info_without_license  # pylint: disable=protected-access

        mock_fetcher_class.return_value = fetcher
        mock_get_vram.side_effect = [24 * 1024**3, 0]

        result = runner.invoke(app, ["model", "info", "test/model"])

        assert result.exit_code == 0

    # ------------------------------------------------------------------
    # Config / display tests
    # ------------------------------------------------------------------

    def test_info_command_context_window_extraction(self, mocker, app, runner):
        """max_position_embeddings is shown; missing key shows 'Unknown'."""
        mock_get_vram = mocker.patch("rzr_aikit.utils.mlib.get_cuda_total_vram")
        mocker.patch(
            "rzr_aikit.utils.connectivity.check_huggingface_connectivity",
            return_value=True,
        )
        mocker.patch(
            "rzr_aikit.utils.DiffusionModelInfoFetcher.DiffusionModelInfoFetcher"
        )

        cases = [
            ({"max_position_embeddings": 2048}, "2048"),
            ({"max_position_embeddings": 4096}, "4096"),
            ({"max_position_embeddings": 8192}, "8192"),
            ({}, "Unknown"),
            ({"hidden_size": 768}, "Unknown"),
        ]

        for config, expected in cases:
            mock_fetcher_class = mocker.patch(
                "rzr_aikit.utils.ModelInfoFetcher.ModelInfoFetcher"
            )
            test_fetcher = mocker.Mock()
            test_fetcher.source = "hub"
            test_fetcher.total_bytes = 1024**3
            test_fetcher.dtype = "float16"
            test_fetcher.config = config
            test_fetcher.quant_config = {}

            model_info = mocker.Mock()
            model_info.pipeline_tag = "text-generation"
            model_info.downloads = 100
            model_info.last_modified = "2024-01-01T00:00:00Z"
            card_data = mocker.Mock()
            card_data.license = "apache-2.0"
            model_info.card_data = card_data
            test_fetcher._model_info = model_info  # pylint: disable=protected-access

            mock_fetcher_class.return_value = test_fetcher
            mock_get_vram.side_effect = [24 * 1024**3, 0]

            result = runner.invoke(app, ["model", "info", "test/model"])

            assert result.exit_code == 0, f"Failed for config={config}"
            assert expected in result.stdout, (
                f"Expected '{expected}' in output for config={config}"
            )

    def test_info_command_quantization_config_display(self, mocker, app, runner):
        """Quantization method from quant_config is shown in Precision field."""
        mock_fetcher_class, mock_get_vram, _ = self._patch_common(mocker)

        quantized_fetcher = mocker.Mock()
        quantized_fetcher.source = "hub"
        quantized_fetcher.total_bytes = 300_000_000
        quantized_fetcher.dtype = "float16"
        quantized_fetcher.config = {"max_position_embeddings": 4096}
        quantized_fetcher.quant_config = {"quant_method": "8bit", "bits": 8}

        model_info = mocker.Mock()
        model_info.pipeline_tag = "text-generation"
        model_info.downloads = 100
        model_info.last_modified = "2024-01-01T00:00:00Z"
        card_data = mocker.Mock()
        card_data.license = "apache-2.0"
        model_info.card_data = card_data
        quantized_fetcher._model_info = model_info  # pylint: disable=protected-access

        mock_fetcher_class.return_value = quantized_fetcher
        mock_get_vram.side_effect = [24 * 1024**3, 0]

        result = runner.invoke(app, ["model", "info", "test/quantized-model"])

        assert result.exit_code == 0
        assert "8bit" in result.stdout

    def test_info_command_full_config_option(self, mocker, app, runner):
        """--full-config flag shows Full Configuration panel with all keys."""
        mock_fetcher_class, mock_get_vram, _ = self._patch_common(mocker)

        detailed_fetcher = mocker.Mock()
        detailed_fetcher.source = "hub"
        detailed_fetcher.total_bytes = 1024**3
        detailed_fetcher.dtype = "float16"
        detailed_fetcher.config = {
            "max_position_embeddings": 4096,
            "hidden_size": 768,
            "num_attention_heads": 12,
            "num_hidden_layers": 24,
            "vocab_size": 50257,
        }
        detailed_fetcher.quant_config = {}

        model_info = mocker.Mock()
        model_info.pipeline_tag = "text-generation"
        model_info.downloads = 100
        model_info.last_modified = "2024-01-01T00:00:00Z"
        card_data = mocker.Mock()
        card_data.license = "apache-2.0"
        model_info.card_data = card_data
        detailed_fetcher._model_info = model_info  # pylint: disable=protected-access

        mock_fetcher_class.return_value = detailed_fetcher
        mock_get_vram.side_effect = [24 * 1024**3, 0]

        result = runner.invoke(app, ["model", "info", "test/model", "--full-config"])

        assert result.exit_code == 0
        assert "Full Configuration" in result.stdout
        assert "hidden_size" in result.stdout
        assert "num_attention_heads" in result.stdout

    def test_info_command_offline_mode(self, mocker, app, runner):
        """Offline mode — hub-specific fields (Downloads) are not shown."""
        mock_fetcher_class, mock_get_vram, _ = self._patch_common(mocker, connected=False)

        offline_fetcher = mocker.Mock()
        offline_fetcher.source = "cache"
        offline_fetcher.total_bytes = 1024**3
        offline_fetcher.dtype = "float16"
        offline_fetcher.config = {"max_position_embeddings": 4096}
        offline_fetcher.quant_config = {}

        mock_fetcher_class.return_value = offline_fetcher
        mock_get_vram.side_effect = [24 * 1024**3, 0]

        result = runner.invoke(app, ["model", "info", "test/model"])

        assert result.exit_code == 0
        assert "Downloads:" not in result.stdout
