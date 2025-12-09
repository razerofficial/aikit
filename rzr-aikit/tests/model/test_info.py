import pytest
from typer.testing import CliRunner


class TestInfoCommand:
    """Test cases for the model info command."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    @pytest.fixture
    def app(self):
        from src.cli import app as cli_app

        return cli_app

    @pytest.fixture
    def mock_fetcher_hub(self, mocker):
        """Mock ModelInfoFetcher for hub source (remote model from HuggingFace)."""
        fetcher = mocker.Mock()
        fetcher.source = "hub"
        fetcher.total_bytes = 600000000  # 600MB
        fetcher.dtype = "float16"  # Uses 'dtype' field from config
        fetcher.config = {
            "max_position_embeddings": 4096,
            "hidden_size": 768,
            "num_attention_heads": 12,
        }
        fetcher.quant_config = {}

        # Mock HuggingFace model info (only available for hub source)
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
        """Mock ModelInfoFetcher for local/cache source (offline model)."""
        fetcher = mocker.Mock()
        fetcher.source = "local"  # Could also be "cache"
        fetcher.total_bytes = 600000000  # 600MB
        fetcher.dtype = "float16"  # Uses 'dtype' field from config
        fetcher.config = {
            "max_position_embeddings": 4096,
            "hidden_size": 768,
            "num_attention_heads": 12,
        }
        fetcher.quant_config = {}
        # No _model_info for local/cache sources

        return fetcher

    def test_info_command_hub_model_nongated(
        self,
        mocker,
        app,
        runner,
        mock_fetcher_hub,
    ):
        """Test model info with non-gated hub model (Qwen/Qwen3-0.6B)."""
        # Mock dependencies
        mock_fetcher_class = mocker.patch("util.ModelInfoFetcher.ModelInfoFetcher")
        mock_get_vram = mocker.patch("util.mlib.get_cuda_total_vram")
        mock_connectivity = mocker.patch(
            "util.connectivity.check_huggingface_connectivity"
        )
        mock_connectivity.return_value = True

        # Setup return values
        mock_fetcher_class.return_value = mock_fetcher_hub
        mock_get_vram.side_effect = [48 * 1024**3, 96 * 1024**3]

        result = runner.invoke(app, ["model", "info", "Qwen/Qwen3-0.6B"])

        assert result.exit_code == 0
        assert "✅ Yes" in result.stdout
        assert "Qwen/Qwen3-0.6B" in result.stdout
        assert "text-generation" in result.stdout  # From hub info
        assert "float16" in result.stdout
        assert "4096" in result.stdout
        assert "12,345" in result.stdout  # Downloads shown for hub model

    def test_info_command_hub_model_gated_with_auth(
        self,
        mocker,
        app,
        runner,
    ):
        """Test model info with gated hub model requiring authentication (google/gemma-3-1b-it)."""
        # Mock dependencies
        mock_fetcher_class = mocker.patch("util.ModelInfoFetcher.ModelInfoFetcher")
        mock_get_vram = mocker.patch("util.mlib.get_cuda_total_vram")
        mock_connectivity = mocker.patch(
            "util.connectivity.check_huggingface_connectivity"
        )
        mock_connectivity.return_value = True

        # Mock HuggingfaceAccessTokenRequired exception on first call
        from src import HuggingfaceAccessTokenRequired

        # Create successful fetcher for second call (after auth)
        gated_fetcher = mocker.Mock()
        gated_fetcher.source = "hub"
        gated_fetcher.total_bytes = 1200000000  # 1.2GB
        gated_fetcher.dtype = "bfloat16"
        gated_fetcher.config = {
            "max_position_embeddings": 8192,
            "hidden_size": 2048,
        }
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

        # First call raises HuggingfaceAccessTokenRequired, second succeeds
        mock_fetcher_class.side_effect = [
            HuggingfaceAccessTokenRequired("Token required"),
            gated_fetcher,
        ]

        # Mock token retrieval
        mock_token_class = mocker.patch("util.HuggingfaceHubToken.HuggingfaceHubToken")
        mock_token_instance = mocker.Mock()
        mock_token_instance.get_access_token.return_value = "hf_test_token_123"
        mock_token_class.return_value = mock_token_instance

        mock_get_vram.side_effect = [24 * 1024**3, 0]

        result = runner.invoke(app, ["model", "info", "google/gemma-3-1b-it"])

        assert result.exit_code == 0
        assert "✅ Yes" in result.stdout
        assert "google/gemma-3-1b-it" in result.stdout
        assert "bfloat16" in result.stdout
        # Verify token was used
        assert mock_token_instance.get_access_token.called

    def test_info_command_hub_model_gated_without_auth(
        self,
        mocker,
        app,
        runner,
    ):
        """Test model info with gated hub model without authentication."""
        # Mock dependencies
        mock_fetcher_class = mocker.patch("util.ModelInfoFetcher.ModelInfoFetcher")
        mock_connectivity = mocker.patch(
            "util.connectivity.check_huggingface_connectivity"
        )
        mock_connectivity.return_value = True

        from src import HuggingfaceAccessTokenRequired

        # Mock token retrieval failure
        mock_token_class = mocker.patch("util.HuggingfaceHubToken.HuggingfaceHubToken")
        mock_token_instance = mocker.Mock()
        mock_token_instance.get_access_token.side_effect = (
            HuggingfaceAccessTokenRequired("No token available")
        )
        mock_token_class.return_value = mock_token_instance

        # Both calls fail
        mock_fetcher_class.side_effect = HuggingfaceAccessTokenRequired(
            "Token required"
        )

        result = runner.invoke(app, ["model", "info", "google/gemma-3-1b-it"])

        # Should fail gracefully
        assert result.exit_code != 0 or "Incompatible model" in result.stdout

    def test_info_command_local_model(
        self,
        mocker,
        app,
        runner,
    ):
        """Test model info with local model (shows hub info when connected)."""
        # Mock dependencies
        mock_fetcher_class = mocker.patch("util.ModelInfoFetcher.ModelInfoFetcher")
        mock_get_vram = mocker.patch("util.mlib.get_cuda_total_vram")
        mock_connectivity = mocker.patch(
            "util.connectivity.check_huggingface_connectivity"
        )
        mock_connectivity.return_value = True  # Local models can also fetch hub info

        # Create local fetcher with hub info
        local_fetcher = mocker.Mock()
        local_fetcher.source = "local"
        local_fetcher.total_bytes = 600000000  # 600MB
        local_fetcher.dtype = "float16"
        local_fetcher.config = {
            "max_position_embeddings": 4096,
            "hidden_size": 768,
            "num_attention_heads": 12,
        }
        local_fetcher.quant_config = {}

        # Local models can also have _model_info if connected
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
        # Local models WITH connection should show hub-specific info
        assert "Downloads:" in result.stdout or "5,000" in result.stdout
        assert "text-generation" in result.stdout
        # And should still show technical info
        assert "float16" in result.stdout
        assert "4096" in result.stdout

    def test_info_command_cached_model(
        self,
        mocker,
        app,
        runner,
    ):
        """Test model info with cached model (source='cache')."""
        # Mock dependencies
        mock_fetcher_class = mocker.patch("util.ModelInfoFetcher.ModelInfoFetcher")
        mock_get_vram = mocker.patch("util.mlib.get_cuda_total_vram")
        mock_connectivity = mocker.patch(
            "util.connectivity.check_huggingface_connectivity"
        )
        mock_connectivity.return_value = False  # Offline

        cached_fetcher = mocker.Mock()
        cached_fetcher.source = "cache"
        cached_fetcher.total_bytes = 600000000
        cached_fetcher.dtype = "float16"
        cached_fetcher.config = {"max_position_embeddings": 4096}
        cached_fetcher.quant_config = {}
        # No _model_info for cache source

        mock_fetcher_class.return_value = cached_fetcher
        mock_get_vram.side_effect = [24 * 1024**3, 0]

        result = runner.invoke(app, ["model", "info", "org/model"])

        assert result.exit_code == 0
        assert "✅ Yes" in result.stdout
        # Cached models behave like local models
        assert "Downloads:" not in result.stdout

    def test_info_command_incompatible_hardware(self, mocker, app, runner):
        """Test model info with incompatible hardware (small GPU, large model)."""
        # Mock dependencies
        mock_fetcher_class = mocker.patch("util.ModelInfoFetcher.ModelInfoFetcher")
        mock_get_vram = mocker.patch("util.mlib.get_cuda_total_vram")
        mock_connectivity = mocker.patch(
            "util.connectivity.check_huggingface_connectivity"
        )
        mock_connectivity.return_value = True

        # Setup large model scenario
        large_fetcher = mocker.Mock()
        large_fetcher.source = "hub"
        large_fetcher.total_bytes = 30 * 1024**3  # 30GB
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
        """Test model info with limited compatibility (4-bit quantization available)."""
        mock_fetcher_class = mocker.patch("util.ModelInfoFetcher.ModelInfoFetcher")
        mock_get_vram = mocker.patch("util.mlib.get_cuda_total_vram")
        mock_connectivity = mocker.patch(
            "util.connectivity.check_huggingface_connectivity"
        )
        mock_connectivity.return_value = True

        model_fetcher = mocker.Mock()
        model_fetcher.source = "hub"
        model_fetcher.total_bytes = 28 * 1024**3  # 28GB
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
        """Test model info when no GPU is available."""
        mock_fetcher_class = mocker.patch("util.ModelInfoFetcher.ModelInfoFetcher")
        mock_get_vram = mocker.patch("util.mlib.get_cuda_total_vram")
        mock_connectivity = mocker.patch(
            "util.connectivity.check_huggingface_connectivity"
        )
        mock_connectivity.return_value = True

        mock_fetcher_class.return_value = mock_fetcher_hub
        mock_get_vram.side_effect = [0, 0]

        result = runner.invoke(app, ["model", "info", "test/model"])

        assert result.exit_code == 0
        assert "❌ No" in result.stdout

    def test_info_command_no_ray_cluster(self, mocker, app, runner, mock_fetcher_hub):
        """Test model info when not connected to Ray cluster."""
        mock_fetcher_class = mocker.patch("util.ModelInfoFetcher.ModelInfoFetcher")
        mock_get_vram = mocker.patch("util.mlib.get_cuda_total_vram")
        mock_connectivity = mocker.patch(
            "util.connectivity.check_huggingface_connectivity"
        )
        mock_connectivity.return_value = True

        mock_fetcher_class.return_value = mock_fetcher_hub
        mock_get_vram.side_effect = [24 * 1024**3, 0]

        result = runner.invoke(app, ["model", "info", "test/model"])

        assert result.exit_code == 0
        assert "Not connected to Ray cluster" in result.stdout

    def test_info_command_ray_discovery_in_progress(
        self, mocker, app, runner, mock_fetcher_hub
    ):
        """Test model info when Ray GPU discovery is in progress."""
        mock_fetcher_class = mocker.patch("util.ModelInfoFetcher.ModelInfoFetcher")
        mock_get_vram = mocker.patch("util.mlib.get_cuda_total_vram")
        mock_connectivity = mocker.patch(
            "util.connectivity.check_huggingface_connectivity"
        )
        mock_connectivity.return_value = True

        mock_fetcher_class.return_value = mock_fetcher_hub
        mock_get_vram.side_effect = [24 * 1024**3, -1]

        result = runner.invoke(app, ["model", "info", "test/model"])

        assert result.exit_code == 0
        assert "GPU discovery is in progress" in result.stdout

    def test_info_command_general_error(self, mocker, app, runner):
        """Test info command with general error."""
        mock_fetcher_class = mocker.patch("util.ModelInfoFetcher.ModelInfoFetcher")
        mock_connectivity = mocker.patch(
            "util.connectivity.check_huggingface_connectivity"
        )
        mock_connectivity.return_value = True

        mock_fetcher_class.side_effect = Exception("General model error")

        result = runner.invoke(app, ["model", "info", "test/model"])

        assert result.exit_code == 0
        assert "Incompatible model" in result.stdout

    def test_info_command_context_window_extraction(self, mocker, app, runner):
        """Test context window extraction - only max_position_embeddings is checked."""
        mock_get_vram = mocker.patch("util.mlib.get_cuda_total_vram")
        mock_connectivity = mocker.patch(
            "util.connectivity.check_huggingface_connectivity"
        )
        mock_connectivity.return_value = True

        configs_to_test = [
            ({"max_position_embeddings": 2048}, "2048"),
            ({"max_position_embeddings": 4096}, "4096"),
            ({"max_position_embeddings": 8192}, "8192"),
            ({}, "Unknown"),  # Missing field
            ({"hidden_size": 768}, "Unknown"),  # No max_position_embeddings
        ]

        for config, expected_output in configs_to_test:
            mock_fetcher_class = mocker.patch("util.ModelInfoFetcher.ModelInfoFetcher")

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

            assert result.exit_code == 0
            assert expected_output in result.stdout

    def test_info_command_license_access_error(self, mocker, app, runner):
        """Test info command when license information is not accessible."""
        mock_fetcher_class = mocker.patch("util.ModelInfoFetcher.ModelInfoFetcher")
        mock_get_vram = mocker.patch("util.mlib.get_cuda_total_vram")
        mock_connectivity = mocker.patch(
            "util.connectivity.check_huggingface_connectivity"
        )
        mock_connectivity.return_value = True

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

    def test_info_command_quantization_config_display(self, mocker, app, runner):
        """Test display of quantization configuration when present."""
        mock_fetcher_class = mocker.patch("util.ModelInfoFetcher.ModelInfoFetcher")
        mock_get_vram = mocker.patch("util.mlib.get_cuda_total_vram")
        mock_connectivity = mocker.patch(
            "util.connectivity.check_huggingface_connectivity"
        )
        mock_connectivity.return_value = True

        quantized_fetcher = mocker.Mock()
        quantized_fetcher.source = "hub"
        quantized_fetcher.total_bytes = 300000000
        quantized_fetcher.config = {"max_position_embeddings": 4096}
        quantized_fetcher.quant_config = {
            "quant_method": "8bit",
            "bits": 8,
        }

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

    def test_info_command_full_config_option(
        self,
        mocker,
        app,
        runner,
    ):
        """Test info command with --full-config option displays complete config."""
        mock_fetcher_class = mocker.patch("util.ModelInfoFetcher.ModelInfoFetcher")
        mock_get_vram = mocker.patch("util.mlib.get_cuda_total_vram")
        mock_connectivity = mocker.patch(
            "util.connectivity.check_huggingface_connectivity"
        )
        mock_connectivity.return_value = True

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
        """Test info command when HuggingFace is not reachable (offline mode)."""
        mock_connectivity = mocker.patch(
            "util.connectivity.check_huggingface_connectivity"
        )
        mock_connectivity.return_value = False

        mock_fetcher_class = mocker.patch("util.ModelInfoFetcher.ModelInfoFetcher")
        mock_get_vram = mocker.patch("util.mlib.get_cuda_total_vram")

        offline_fetcher = mocker.Mock()
        offline_fetcher.source = "cache"  # Offline means using cache
        offline_fetcher.total_bytes = 1 * 1024**3
        offline_fetcher.dtype = "float16"
        offline_fetcher.config = {"max_position_embeddings": 4096}
        offline_fetcher.quant_config = {}
        # No _model_info in offline mode

        mock_fetcher_class.return_value = offline_fetcher
        mock_get_vram.side_effect = [24 * 1024**3, 0]

        result = runner.invoke(app, ["model", "info", "test/model"])

        assert result.exit_code == 0
        # In offline mode, hub-specific info won't be shown
        assert "Downloads:" not in result.stdout
