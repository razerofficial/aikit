import pytest
from typer.testing import CliRunner


class TestListCommand:
    """Test cases for the model list command."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    @pytest.fixture
    def app(self):
        from src.cli import app as cli_app

        return cli_app

    @pytest.fixture
    def mock_gpu_discovery(self, mocker):
        """Mock GPU discovery to simulate local and distributed GPU environments."""
        mock_discover = mocker.patch("gpu_discovery.discover.discover_gpus")

        # Default: 2 local GPUs (24GB each) + 4 distributed GPUs (24GB each)
        mock_discover.return_value = {
            "local": [
                {"total_memory": 24 * 1024**3, "name": "NVIDIA RTX 4090"},
                {"total_memory": 24 * 1024**3, "name": "NVIDIA RTX 4090"},
            ],
            "ray": [
                {
                    "node_id": "node1",
                    "gpus": [
                        {"total_memory": 24 * 1024**3, "name": "NVIDIA RTX 4090"},
                        {"total_memory": 24 * 1024**3, "name": "NVIDIA RTX 4090"},
                    ],
                },
                {
                    "node_id": "node2",
                    "gpus": [
                        {"total_memory": 24 * 1024**3, "name": "NVIDIA RTX 4090"},
                        {"total_memory": 24 * 1024**3, "name": "NVIDIA RTX 4090"},
                    ],
                },
            ],
        }

        return mock_discover

    @pytest.fixture
    def mock_repo_with_weights(self, mocker):
        """Mock repository with model weights."""
        repo = mocker.Mock()
        repo.repo_type = "model"
        repo.repo_id = "Qwen/Qwen3-0.6B"
        repo.size_on_disk = 600000000

        # Mock snapshots with weight files
        snapshot = mocker.Mock()
        file_info = mocker.Mock()
        file_info.file_name = "model.safetensors"
        snapshot.files = [file_info]
        repo.snapshots = [snapshot]

        return repo

    @pytest.fixture
    def mock_repo_without_weights(self, mocker):
        """Mock repository without model weights (config only)."""
        repo = mocker.Mock()
        repo.repo_type = "model"
        repo.repo_id = "test/config-only"
        repo.size_on_disk = 1000000

        # Mock snapshots with only config files
        snapshot = mocker.Mock()
        file_info = mocker.Mock()
        file_info.file_name = "config.json"
        snapshot.files = [file_info]
        repo.snapshots = [snapshot]

        return repo

    @pytest.fixture
    def mock_dataset_repo(self, mocker):
        """Mock dataset repository (should be filtered out)."""
        repo = mocker.Mock()
        repo.repo_type = "dataset"
        repo.repo_id = "test/dataset"
        repo.size_on_disk = 5000000
        return repo

    def test_list_command_success(
        self, mocker, app, runner, mock_gpu_discovery, mock_repo_with_weights
    ):
        """Test successful model listing with compatible models."""
        # Setup mocks
        mock_scan_cache = mocker.patch("huggingface_hub.scan_cache_dir")
        mock_get_vram = mocker.patch("util.mlib.get_cuda_total_vram")
        mock_connectivity = mocker.patch(
            "util.connectivity.check_huggingface_connectivity"
        )
        mock_connectivity.return_value = True

        # Mock ModelInfoFetcher
        mock_fetcher_class = mocker.patch("util.ModelInfoFetcher.ModelInfoFetcher")
        mock_fetcher = mocker.Mock()
        mock_fetcher.total_bytes = 600000000  # 600MB
        mock_fetcher.dtype = "float16"
        mock_fetcher.quant_config = {}
        mock_fetcher_class.return_value = mock_fetcher

        cache_info = mocker.Mock()
        cache_info.repos = [mock_repo_with_weights]
        mock_scan_cache.return_value = cache_info
        # 600MB * 1.2 = 720MB < 48GB (local) and < 96GB (distributed)
        mock_get_vram.side_effect = [
            48 * 1024**3,
            96 * 1024**3,
        ]  # Local, then distributed

        result = runner.invoke(app, ["model", "list"])

        assert result.exit_code == 0
        assert "Qwen/Qwen3-0.6B" in result.stdout
        assert "✅ Yes" in result.stdout  # Compatible
        # Verify get_vram was called for both local and distributed
        assert mock_get_vram.call_count >= 1

    def test_list_command_incompatible_model(
        self, mocker, app, runner, mock_gpu_discovery, mock_repo_with_weights
    ):
        """Test listing with incompatible model."""
        mock_scan_cache = mocker.patch("huggingface_hub.scan_cache_dir")
        mock_get_vram = mocker.patch("util.mlib.get_cuda_total_vram")
        mock_connectivity = mocker.patch(
            "util.connectivity.check_huggingface_connectivity"
        )
        mock_connectivity.return_value = True

        # Mock ModelInfoFetcher with large model that won't fit
        mock_fetcher_class = mocker.patch("util.ModelInfoFetcher.ModelInfoFetcher")
        mock_fetcher = mocker.Mock()
        mock_fetcher.total_bytes = 100 * 1024**3  # 100GB model
        mock_fetcher.dtype = "float32"
        mock_fetcher.quant_config = {}
        mock_fetcher_class.return_value = mock_fetcher

        cache_info = mocker.Mock()
        cache_info.repos = [mock_repo_with_weights]
        mock_scan_cache.return_value = cache_info
        # 100GB * 1.2 = 120GB > 48GB (incompatible)
        # 100GB * 1.2 * (4/32) = 15GB < 48GB (would be limited, but let's test incompatible)
        # Actually, let's use smaller GPU memory
        mock_get_vram.side_effect = [
            8 * 1024**3,
            16 * 1024**3,
        ]  # Too small even with quantization

        result = runner.invoke(app, ["model", "list"])

        assert result.exit_code == 0
        assert "Qwen/Qwen3-0.6B" in result.stdout
        assert "❌ No" in result.stdout  # Incompatible

    def test_list_command_limited_compatibility(
        self, mocker, app, runner, mock_gpu_discovery, mock_repo_with_weights
    ):
        """Test listing with limited compatibility (4-bit quantization available)."""
        mock_scan_cache = mocker.patch("huggingface_hub.scan_cache_dir")
        mock_get_vram = mocker.patch("util.mlib.get_cuda_total_vram")
        mock_connectivity = mocker.patch(
            "util.connectivity.check_huggingface_connectivity"
        )
        mock_connectivity.return_value = True

        # Mock ModelInfoFetcher
        mock_fetcher_class = mocker.patch("util.ModelInfoFetcher.ModelInfoFetcher")
        mock_fetcher = mocker.Mock()
        mock_fetcher.total_bytes = 50 * 1024**3  # 50GB model
        mock_fetcher.dtype = "float16"
        mock_fetcher.quant_config = {}
        mock_fetcher_class.return_value = mock_fetcher

        cache_info = mocker.Mock()
        cache_info.repos = [mock_repo_with_weights]
        mock_scan_cache.return_value = cache_info
        # 50GB * 1.2 = 60GB > 48GB (incompatible)
        # 50GB * 1.2 * (4/16) = 15GB < 48GB (compatible with 4-bit - Limited)
        mock_get_vram.side_effect = [48 * 1024**3, 96 * 1024**3]

        result = runner.invoke(app, ["model", "list"])

        assert result.exit_code == 0
        assert "Qwen/Qwen3-0.6B" in result.stdout
        assert "🟡 Limited" in result.stdout  # Limited compatibility

    def test_list_command_model_compatibility_error(
        self, mocker, app, runner, mock_gpu_discovery, mock_repo_with_weights
    ):
        """Test listing when model compatibility check fails."""
        mock_scan_cache = mocker.patch("huggingface_hub.scan_cache_dir")
        mock_fetcher_class = mocker.patch("util.ModelInfoFetcher.ModelInfoFetcher")
        mock_connectivity = mocker.patch(
            "util.connectivity.check_huggingface_connectivity"
        )
        mock_connectivity.return_value = True

        cache_info = mocker.Mock()
        cache_info.repos = [mock_repo_with_weights]
        mock_scan_cache.return_value = cache_info
        mock_fetcher_class.side_effect = Exception("Model loading error")

        result = runner.invoke(app, ["model", "list"])

        assert result.exit_code == 0
        assert "Skipping model" in result.stdout
        assert "Qwen/Qwen3-0.6B" in result.stdout

    def test_list_command_no_cached_models(self, mocker, app, runner):
        """Test listing when no models are cached."""
        # Mock GPU discovery
        mock_discover = mocker.patch("gpu_discovery.discover.discover_gpus")
        mock_discover.return_value = {
            "local": [
                {"total_memory": 24 * 1024**3, "name": "NVIDIA RTX 4090"},
            ],
            "ray": [],
        }

        mock_scan_cache = mocker.patch("huggingface_hub.scan_cache_dir")

        cache_info = mocker.Mock()
        cache_info.repos = []
        mock_scan_cache.return_value = cache_info

        result = runner.invoke(app, ["model", "list"])

        assert result.exit_code == 0
        # Should complete successfully even with no models

    def test_list_command_filters_non_model_repos(
        self, mocker, app, runner, mock_dataset_repo
    ):
        """Test that non-model repositories are filtered out."""
        # Mock GPU discovery
        mock_discover = mocker.patch("gpu_discovery.discover.discover_gpus")
        mock_discover.return_value = {
            "local": [
                {"total_memory": 24 * 1024**3, "name": "NVIDIA RTX 4090"},
            ],
            "ray": [],
        }

        mock_scan_cache = mocker.patch("huggingface_hub.scan_cache_dir")

        cache_info = mocker.Mock()
        cache_info.repos = [mock_dataset_repo]
        mock_scan_cache.return_value = cache_info

        result = runner.invoke(app, ["model", "list"])

        assert result.exit_code == 0
        assert "test/dataset" not in result.stdout

    def test_list_command_filters_repos_without_weights(
        self, mocker, app, runner, mock_repo_without_weights
    ):
        """Test that repositories without model weights are filtered out."""
        # Mock GPU discovery
        mock_discover = mocker.patch("gpu_discovery.discover.discover_gpus")
        mock_discover.return_value = {
            "local": [
                {"total_memory": 24 * 1024**3, "name": "NVIDIA RTX 4090"},
            ],
            "ray": [],
        }

        mock_scan_cache = mocker.patch("huggingface_hub.scan_cache_dir")

        cache_info = mocker.Mock()
        cache_info.repos = [mock_repo_without_weights]
        mock_scan_cache.return_value = cache_info

        result = runner.invoke(app, ["model", "list"])

        assert result.exit_code == 0
        assert "test/config-only" not in result.stdout

    def test_list_command_multiple_models(self, mocker, app, runner):
        """Test listing multiple models with different compatibility."""
        # Mock GPU discovery
        mock_discover = mocker.patch("gpu_discovery.discover.discover_gpus")
        mock_discover.return_value = {
            "local": [
                {"total_memory": 24 * 1024**3, "name": "NVIDIA RTX 4090"},
            ],
            "ray": [],
        }

        mock_scan_cache = mocker.patch("huggingface_hub.scan_cache_dir")
        mock_get_vram = mocker.patch("util.mlib.get_cuda_total_vram")

        # Create multiple test repos
        repo1 = mocker.Mock()
        repo1.repo_type = "model"
        repo1.repo_id = "model/compatible"
        repo1.size_on_disk = 500000000
        repo1.snapshots = [mocker.Mock()]
        repo1.snapshots[0].files = [mocker.Mock()]
        repo1.snapshots[0].files[0].file_name = "model.safetensors"

        repo2 = mocker.Mock()
        repo2.repo_type = "model"
        repo2.repo_id = "model/incompatible"
        repo2.size_on_disk = 70000000000  # 70GB
        repo2.snapshots = [mocker.Mock()]
        repo2.snapshots[0].files = [mocker.Mock()]
        repo2.snapshots[0].files[0].file_name = "pytorch_model.bin"

        cache_info = mocker.Mock()
        cache_info.repos = [repo1, repo2]
        mock_scan_cache.return_value = cache_info

        # Mock ModelInfoFetcher to return different sizes
        mock_fetcher_class = mocker.patch("util.ModelInfoFetcher.ModelInfoFetcher")

        def fetcher_side_effect(model_id):
            fetcher = mocker.Mock()
            if "compatible" in model_id:
                fetcher.total_bytes = 1 * 1024**3  # 1GB
                fetcher.dtype = "float16"
            else:
                fetcher.total_bytes = 100 * 1024**3  # 100GB
                fetcher.dtype = "float32"
            fetcher.quant_config = {}
            return fetcher

        mock_fetcher_class.side_effect = fetcher_side_effect
        mock_get_vram.side_effect = [
            24 * 1024**3,
            0,
        ] * 2  # For each model: local, distributed

        result = runner.invoke(app, ["model", "list"])

        assert result.exit_code == 0
        assert "model/compatible" in result.stdout
        assert "model/incompatible" in result.stdout
        # Should show different compatibility statuses

    def test_list_command_scan_cache_error(self, mocker, app, runner):
        """Test listing when cache scanning fails."""
        # Mock GPU discovery
        mock_discover = mocker.patch("gpu_discovery.discover.discover_gpus")
        mock_discover.return_value = {
            "local": [
                {"total_memory": 24 * 1024**3, "name": "NVIDIA RTX 4090"},
            ],
            "ray": [],
        }

        mock_scan_cache = mocker.patch("huggingface_hub.scan_cache_dir")
        mock_scan_cache.side_effect = Exception("Cache directory not accessible")

        result = runner.invoke(app, ["model", "list"])

        assert result.exit_code == 0
        # Should handle the error gracefully

    def test_list_command_no_gpu(self, mocker, app, runner):
        """Test listing when no GPU is available."""
        # Mock no GPU environment
        mock_discover = mocker.patch("gpu_discovery.discover.discover_gpus")
        mock_discover.return_value = {
            "local": [],
            "ray": [],
        }

        mock_scan_cache = mocker.patch("huggingface_hub.scan_cache_dir")
        mock_get_vram = mocker.patch("util.mlib.get_cuda_total_vram")

        # Mock ModelInfoFetcher
        mock_fetcher_class = mocker.patch("util.ModelInfoFetcher.ModelInfoFetcher")

        repo = mocker.Mock()
        repo.repo_type = "model"
        repo.repo_id = "test/model"
        repo.size_on_disk = 1000000
        repo.snapshots = [mocker.Mock()]
        repo.snapshots[0].files = [mocker.Mock()]
        repo.snapshots[0].files[0].file_name = "model.bin"

        cache_info = mocker.Mock()
        cache_info.repos = [repo]
        mock_scan_cache.return_value = cache_info

        mock_fetcher = mocker.Mock()
        mock_fetcher.total_bytes = 1 * 1024**3
        mock_fetcher.dtype = "float16"
        mock_fetcher.quant_config = {}
        mock_fetcher_class.return_value = mock_fetcher

        mock_get_vram.side_effect = [0, 0]  # No GPU memory

        result = runner.invoke(app, ["model", "list"])

        assert result.exit_code == 0
        assert "❌ No" in result.stdout  # Should be incompatible without GPU

    def test_has_weights_with_safetensors(self, mocker):
        """Test has_weights function with safetensors files."""
        from src.model.list import has_weights

        repo = mocker.Mock()
        snapshot = mocker.Mock()
        file_info = mocker.Mock()
        file_info.file_name = "model.safetensors"
        snapshot.files = [file_info]
        repo.snapshots = [snapshot]

        assert has_weights(repo) == True

    def test_has_weights_with_pytorch_bin(self, mocker):
        """Test has_weights function with pytorch bin files."""
        from src.model.list import has_weights

        repo = mocker.Mock()
        snapshot = mocker.Mock()
        file_info = mocker.Mock()
        file_info.file_name = "pytorch_model.bin"
        snapshot.files = [file_info]
        repo.snapshots = [snapshot]

        assert has_weights(repo) == True

    def test_has_weights_with_pt_files(self, mocker):
        """Test has_weights function with .pt files."""
        from src.model.list import has_weights

        repo = mocker.Mock()
        snapshot = mocker.Mock()
        file_info = mocker.Mock()
        file_info.file_name = "model.pt"
        snapshot.files = [file_info]
        repo.snapshots = [snapshot]

        assert has_weights(repo) == True

    def test_has_weights_without_weight_files(self, mocker):
        """Test has_weights function with only config files."""
        from src.model.list import has_weights

        repo = mocker.Mock()
        snapshot = mocker.Mock()
        file_info = mocker.Mock()
        file_info.file_name = "config.json"
        snapshot.files = [file_info]
        repo.snapshots = [snapshot]

        assert has_weights(repo) == False

    def test_has_weights_filters_config_files(self, mocker):
        """Test has_weights function filters out config and tokenizer files."""
        from src.model.list import has_weights

        repo = mocker.Mock()
        snapshot = mocker.Mock()

        # Files that should be ignored
        config_file = mocker.Mock()
        config_file.file_name = "config.json"
        tokenizer_file = mocker.Mock()
        tokenizer_file.file_name = "tokenizer.json"

        snapshot.files = [config_file, tokenizer_file]
        repo.snapshots = [snapshot]

        assert has_weights(repo) == False

    def test_has_weights_fallback_filesystem_scan(self, mocker):
        """Test has_weights fallback to filesystem scan when no snapshots."""
        from src.model.list import has_weights

        mock_expanduser = mocker.patch("os.path.expanduser")
        mock_walk = mocker.patch("os.walk")
        mock_isdir = mocker.patch("os.path.isdir")

        repo = mocker.Mock()
        repo.snapshots = []  # No snapshots, should trigger fallback
        repo.repo_id = "test/model"

        mock_expanduser.return_value = "/home/user/.cache/huggingface/hub"
        mock_isdir.return_value = True
        mock_walk.return_value = [
            ("/cache/path", [], ["model.safetensors", "config.json"])
        ]

        assert has_weights(repo) == True

    def test_has_weights_fallback_no_cache_dir(self, mocker):
        """Test has_weights fallback when cache directory doesn't exist."""
        from src.model.list import has_weights

        mock_expanduser = mocker.patch("os.path.expanduser")
        mock_isdir = mocker.patch("os.path.isdir")

        repo = mocker.Mock()
        repo.snapshots = []
        repo.repo_id = "test/model"

        mock_expanduser.return_value = "/home/user/.cache/huggingface/hub"
        mock_isdir.return_value = False  # Cache directory doesn't exist

        assert has_weights(repo) == False

    def test_list_command_model_sorting(self, mocker, app, runner):
        """Test that models are sorted alphabetically."""
        # Mock GPU discovery
        mock_discover = mocker.patch("gpu_discovery.discover.discover_gpus")
        mock_discover.return_value = {
            "local": [
                {"total_memory": 24 * 1024**3, "name": "NVIDIA RTX 4090"},
            ],
            "ray": [],
        }

        mock_scan_cache = mocker.patch("huggingface_hub.scan_cache_dir")
        mock_get_vram = mocker.patch("util.mlib.get_cuda_total_vram")

        # Mock ModelInfoFetcher
        mock_fetcher_class = mocker.patch("util.ModelInfoFetcher.ModelInfoFetcher")

        # Create repos in non-alphabetical order
        repo_z = mocker.Mock()
        repo_z.repo_type = "model"
        repo_z.repo_id = "zzz/model"
        repo_z.size_on_disk = 1000000
        repo_z.snapshots = [mocker.Mock()]
        repo_z.snapshots[0].files = [mocker.Mock()]
        repo_z.snapshots[0].files[0].file_name = "model.bin"

        repo_a = mocker.Mock()
        repo_a.repo_type = "model"
        repo_a.repo_id = "aaa/model"
        repo_a.size_on_disk = 2000000
        repo_a.snapshots = [mocker.Mock()]
        repo_a.snapshots[0].files = [mocker.Mock()]
        repo_a.snapshots[0].files[0].file_name = "model.bin"

        cache_info = mocker.Mock()
        cache_info.repos = [repo_z, repo_a]  # Z before A
        mock_scan_cache.return_value = cache_info

        mock_fetcher = mocker.Mock()
        mock_fetcher.total_bytes = 1 * 1024**3
        mock_fetcher.dtype = "float16"
        mock_fetcher.quant_config = {}
        mock_fetcher_class.return_value = mock_fetcher

        mock_get_vram.side_effect = [24 * 1024**3, 0] * 2  # For each model

        result = runner.invoke(app, ["model", "list"])

        assert result.exit_code == 0
        # Check that aaa/model appears before zzz/model in output
        output_lines = result.stdout.split("\n")
        aaa_line = next(
            (i for i, line in enumerate(output_lines) if "aaa/model" in line), -1
        )
        zzz_line = next(
            (i for i, line in enumerate(output_lines) if "zzz/model" in line), -1
        )

        if aaa_line != -1 and zzz_line != -1:
            assert aaa_line < zzz_line

    def test_list_command_offline_mode(
        self, mocker, app, runner, mock_repo_with_weights
    ):
        """Test listing when HuggingFace is not reachable (offline mode)."""
        # Mock GPU discovery
        mock_discover = mocker.patch("gpu_discovery.discover.discover_gpus")
        mock_discover.return_value = {
            "local": [
                {"total_memory": 24 * 1024**3, "name": "NVIDIA RTX 4090"},
            ],
            "ray": [],
        }

        # Mock connectivity check to return False (offline)
        mock_connectivity = mocker.patch(
            "util.connectivity.check_huggingface_connectivity"
        )
        mock_connectivity.return_value = False

        mock_scan_cache = mocker.patch("huggingface_hub.scan_cache_dir")
        mock_get_vram = mocker.patch("util.mlib.get_cuda_total_vram")

        # Mock ModelInfoFetcher
        mock_fetcher_class = mocker.patch("util.ModelInfoFetcher.ModelInfoFetcher")
        mock_fetcher = mocker.Mock()
        mock_fetcher.total_bytes = 1 * 1024**3
        mock_fetcher.dtype = "float16"
        mock_fetcher.quant_config = {}
        mock_fetcher_class.return_value = mock_fetcher

        cache_info = mocker.Mock()
        cache_info.repos = [mock_repo_with_weights]
        mock_scan_cache.return_value = cache_info
        mock_get_vram.side_effect = [24 * 1024**3, 0]

        result = runner.invoke(app, ["model", "list"])

        assert result.exit_code == 0
        # In offline mode, ModelInfoFetcher should be called with allow_internet=False
        mock_fetcher_class.assert_called_with(
            mock_repo_with_weights.repo_id, allow_internet=False
        )

    def test_list_command_fit_model_size(self, mocker, app, runner):
        """Test listing model that fits perfectly without quantization (✅ Yes)."""
        # Mock GPU discovery with 24GB GPU
        mock_discover = mocker.patch("gpu_discovery.discover.discover_gpus")
        mock_discover.return_value = {
            "local": [
                {"total_memory": 24 * 1024**3, "name": "NVIDIA RTX 4090"},
            ],
            "ray": [],
        }

        mock_scan_cache = mocker.patch("huggingface_hub.scan_cache_dir")
        mock_get_vram = mocker.patch("util.mlib.get_cuda_total_vram")
        mock_connectivity = mocker.patch(
            "util.connectivity.check_huggingface_connectivity"
        )
        mock_connectivity.return_value = True

        # Create test repo
        repo = mocker.Mock()
        repo.repo_type = "model"
        repo.repo_id = "small/model"
        repo.size_on_disk = 10 * 1024**3
        repo.snapshots = [mocker.Mock()]
        repo.snapshots[0].files = [mocker.Mock()]
        repo.snapshots[0].files[0].file_name = "model.safetensors"

        # Mock ModelInfoFetcher - 10GB * 1.2 = 12GB < 24GB -> ✅ Yes
        mock_fetcher_class = mocker.patch("util.ModelInfoFetcher.ModelInfoFetcher")
        mock_fetcher = mocker.Mock()
        mock_fetcher.total_bytes = 10 * 1024**3  # 10GB
        mock_fetcher.dtype = "float16"
        mock_fetcher.quant_config = {}
        mock_fetcher_class.return_value = mock_fetcher

        cache_info = mocker.Mock()
        cache_info.repos = [repo]
        mock_scan_cache.return_value = cache_info
        mock_get_vram.side_effect = [24 * 1024**3, 0]

        result = runner.invoke(app, ["model", "list"])

        assert result.exit_code == 0
        assert "small/model" in result.stdout
        assert "✅ Yes" in result.stdout

    def test_list_command_limited_model_size(self, mocker, app, runner):
        """Test listing model that needs 4-bit quantization (🟡 Limited)."""
        # Mock GPU discovery with 16GB GPU
        mock_discover = mocker.patch("gpu_discovery.discover.discover_gpus")
        mock_discover.return_value = {
            "local": [
                {"total_memory": 16 * 1024**3, "name": "NVIDIA RTX 3080"},
            ],
            "ray": [],
        }

        mock_scan_cache = mocker.patch("huggingface_hub.scan_cache_dir")
        mock_get_vram = mocker.patch("util.mlib.get_cuda_total_vram")
        mock_connectivity = mocker.patch(
            "util.connectivity.check_huggingface_connectivity"
        )
        mock_connectivity.return_value = True

        # Create test repo
        repo = mocker.Mock()
        repo.repo_type = "model"
        repo.repo_id = "limited/model"
        repo.size_on_disk = 28 * 1024**3
        repo.snapshots = [mocker.Mock()]
        repo.snapshots[0].files = [mocker.Mock()]
        repo.snapshots[0].files[0].file_name = "model.safetensors"

        # Mock ModelInfoFetcher
        # 28GB * 1.2 = 33.6GB > 16GB (won't fit)
        # But: 28GB * 1.2 * (4/16) = 8.4GB < 16GB (fits with 4-bit) -> 🟡 Limited
        mock_fetcher_class = mocker.patch("util.ModelInfoFetcher.ModelInfoFetcher")
        mock_fetcher = mocker.Mock()
        mock_fetcher.total_bytes = 28 * 1024**3  # 28GB
        mock_fetcher.dtype = "float16"
        mock_fetcher.quant_config = {}
        mock_fetcher_class.return_value = mock_fetcher

        cache_info = mocker.Mock()
        cache_info.repos = [repo]
        mock_scan_cache.return_value = cache_info
        mock_get_vram.side_effect = [16 * 1024**3, 0]

        result = runner.invoke(app, ["model", "list"])

        assert result.exit_code == 0
        assert "limited/model" in result.stdout
        assert "🟡 Limited" in result.stdout
        assert "4-bit on-the-fly quantization" in result.stdout

    def test_list_command_incompatible_model_size(self, mocker, app, runner):
        """Test listing model that doesn't fit even with 4-bit quantization (❌ No)."""
        # Mock GPU discovery with 8GB GPU
        mock_discover = mocker.patch("gpu_discovery.discover.discover_gpus")
        mock_discover.return_value = {
            "local": [
                {"total_memory": 8 * 1024**3, "name": "NVIDIA GTX 1070"},
            ],
            "ray": [],
        }

        mock_scan_cache = mocker.patch("huggingface_hub.scan_cache_dir")
        mock_get_vram = mocker.patch("util.mlib.get_cuda_total_vram")
        mock_connectivity = mocker.patch(
            "util.connectivity.check_huggingface_connectivity"
        )
        mock_connectivity.return_value = True

        # Create test repo
        repo = mocker.Mock()
        repo.repo_type = "model"
        repo.repo_id = "huge/model"
        repo.size_on_disk = 70 * 1024**3
        repo.snapshots = [mocker.Mock()]
        repo.snapshots[0].files = [mocker.Mock()]
        repo.snapshots[0].files[0].file_name = "model.safetensors"

        # Mock ModelInfoFetcher
        # 70GB * 1.2 = 84GB > 8GB (won't fit)
        # Even with 4-bit: 70GB * 1.2 * (4/16) = 21GB > 8GB (still won't fit) -> ❌ No
        mock_fetcher_class = mocker.patch("util.ModelInfoFetcher.ModelInfoFetcher")
        mock_fetcher = mocker.Mock()
        mock_fetcher.total_bytes = 70 * 1024**3  # 70GB
        mock_fetcher.dtype = "float16"
        mock_fetcher.quant_config = {}
        mock_fetcher_class.return_value = mock_fetcher

        cache_info = mocker.Mock()
        cache_info.repos = [repo]
        mock_scan_cache.return_value = cache_info
        mock_get_vram.side_effect = [8 * 1024**3, 0]

        result = runner.invoke(app, ["model", "list"])

        assert result.exit_code == 0
        assert "huge/model" in result.stdout
        assert "❌ No" in result.stdout
