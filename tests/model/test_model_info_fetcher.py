import sys
import json
import pytest
from unittest.mock import MagicMock

sys.modules.setdefault("transformers", MagicMock())


class TestExtractDtype:
    """Unit tests for _extract_dtype."""

    @pytest.fixture
    def extract(self):
        from rzr_aikit.utils.ModelInfoFetcher import _extract_dtype
        return _extract_dtype

    def test_torch_dtype_string(self, extract):
        assert extract({"torch_dtype": "bfloat16"}) == "bfloat16"

    def test_dtype_string(self, extract):
        assert extract({"dtype": "float16"}) == "float16"

    def test_torch_dtype_takes_priority_over_dtype(self, extract):
        # torch_dtype is checked first
        assert extract({"torch_dtype": "bfloat16", "dtype": "float32"}) == "bfloat16"

    def test_uppercase_string_is_lowercased(self, extract):
        assert extract({"torch_dtype": "BFloat16"}) == "bfloat16"

    def test_missing_both_keys_returns_float32(self, extract):
        assert extract({}) == "float32"

    def test_none_value_returns_float32(self, extract):
        assert extract({"torch_dtype": None}) == "float32"

    def test_torch_dtype_object(self, extract):
        """New transformers: AutoConfig.to_dict() returns torch.dtype object."""
        class FakeDtype:
            def __str__(self):
                return "torch.bfloat16"

        assert extract({"torch_dtype": FakeDtype()}) == "bfloat16"

    def test_torch_float16_object(self, extract):
        class FakeDtype:
            def __str__(self):
                return "torch.float16"

        assert extract({"torch_dtype": FakeDtype()}) == "float16"

    def test_torch_float32_object(self, extract):
        class FakeDtype:
            def __str__(self):
                return "torch.float32"

        assert extract({"torch_dtype": FakeDtype()}) == "float32"


class TestNormalizeMistralParams:
    """Unit tests for _normalize_mistral_params."""

    @pytest.fixture
    def normalize(self):
        from rzr_aikit.utils.ModelInfoFetcher import _normalize_mistral_params
        return _normalize_mistral_params

    def test_missing_dtype_defaults_to_bfloat16(self, normalize):
        result = normalize({"dim": 4096, "n_layers": 32})
        assert result["torch_dtype"] == "bfloat16"

    def test_existing_dtype_is_preserved(self, normalize):
        result = normalize({"dim": 4096, "dtype": "float16"})
        assert result["torch_dtype"] == "float16"

    def test_quantization_dict_is_passed_through(self, normalize):
        quant = {"type": "fp8", "bits": 8}
        result = normalize({"dim": 4096, "quantization": quant})
        assert result["quantization_config"] == quant

    def test_missing_quantization_yields_empty_dict(self, normalize):
        result = normalize({"dim": 4096})
        assert result["quantization_config"] == {}

    def test_non_dict_quantization_yields_empty_dict(self, normalize):
        result = normalize({"dim": 4096, "quantization": "fp8"})
        assert result["quantization_config"] == {}

    def test_original_keys_are_preserved(self, normalize):
        result = normalize({"dim": 4096, "n_layers": 32, "vocab_size": 32000})
        assert result["dim"] == 4096
        assert result["n_layers"] == 32
        assert result["vocab_size"] == 32000


class TestModelInfoFetcherMistralRemote:
    """Integration-style tests: config.json 404s, params.json returns Mistral payload."""

    VOXTRAL_PARAMS = {
        "dim": 4096,
        "n_layers": 32,
        "n_heads": 32,
        "vocab_size": 32768,
    }

    def _make_response(self, status_code, json_data=None):
        resp = MagicMock()
        resp.status_code = status_code
        if json_data is not None:
            resp.json.return_value = json_data
        resp.raise_for_status = MagicMock()
        return resp

    def test_hub_source_falls_back_to_params_json(self, mocker):
        """When config.json returns 404 and params.json succeeds, fetcher is populated."""
        from rzr_aikit.utils.ModelInfoFetcher import ModelInfoFetcher

        mock_hfapi = mocker.patch("rzr_aikit.utils.ModelInfoFetcher.HfApi")
        mock_model_info = MagicMock()
        sibling = MagicMock()
        sibling.rfilename = "consolidated.safetensors"
        sibling.size = 8_000_000_000
        mock_model_info.siblings = [sibling]
        mock_hfapi.return_value.model_info.return_value = mock_model_info

        mocker.patch(
            "rzr_aikit.utils.ModelInfoFetcher.requests.get",
            side_effect=[self._make_response(404), self._make_response(200, self.VOXTRAL_PARAMS)],
        )
        mocker.patch(
            "rzr_aikit.utils.ModelInfoFetcher.snapshot_download",
            side_effect=Exception("not cached"),
        )

        fetcher = ModelInfoFetcher("mistralai/Voxtral-Mini-3B-2507", allow_internet=True)

        assert fetcher.dtype == "bfloat16"
        assert fetcher.quant_config == {}
        assert fetcher.total_bytes == 8_000_000_000
        assert fetcher.config["dim"] == 4096

    def test_hub_source_params_json_with_quantization(self, mocker):
        """Mistral params.json with quantization field populates quant_config."""
        from rzr_aikit.utils.ModelInfoFetcher import ModelInfoFetcher

        params = {**self.VOXTRAL_PARAMS, "quantization": {"type": "fp8", "bits": 8}}

        mock_hfapi = mocker.patch("rzr_aikit.utils.ModelInfoFetcher.HfApi")
        mock_model_info = MagicMock()
        mock_model_info.siblings = []
        mock_hfapi.return_value.model_info.return_value = mock_model_info

        mocker.patch(
            "rzr_aikit.utils.ModelInfoFetcher.requests.get",
            side_effect=[self._make_response(404), self._make_response(200, params)],
        )
        mocker.patch(
            "rzr_aikit.utils.ModelInfoFetcher.snapshot_download",
            side_effect=Exception("not cached"),
        )

        fetcher = ModelInfoFetcher("mistralai/Voxtral-FP8", allow_internet=True)

        assert fetcher.quant_config == {"type": "fp8", "bits": 8}

    def test_hub_source_raises_when_both_configs_missing(self, mocker):
        """When both config.json and params.json return 404, FileNotFoundError is raised."""
        from rzr_aikit.utils.ModelInfoFetcher import ModelInfoFetcher

        mock_hfapi = mocker.patch("rzr_aikit.utils.ModelInfoFetcher.HfApi")
        mock_model_info = MagicMock()
        mock_model_info.siblings = []
        mock_hfapi.return_value.model_info.return_value = mock_model_info

        mocker.patch(
            "rzr_aikit.utils.ModelInfoFetcher.requests.get",
            return_value=self._make_response(404),
        )
        mocker.patch(
            "rzr_aikit.utils.ModelInfoFetcher.snapshot_download",
            side_effect=Exception("not cached"),
        )

        with pytest.raises(FileNotFoundError):
            ModelInfoFetcher("unknown/repo-no-config", allow_internet=True)

    def test_hub_source_config_json_takes_priority_over_params_json(self, mocker):
        """config.json is tried first; params.json is never fetched when config.json succeeds."""
        from rzr_aikit.utils.ModelInfoFetcher import ModelInfoFetcher

        hf_config = {"torch_dtype": "float16", "max_position_embeddings": 4096}

        mock_hfapi = mocker.patch("rzr_aikit.utils.ModelInfoFetcher.HfApi")
        mock_model_info = MagicMock()
        mock_model_info.siblings = []
        mock_hfapi.return_value.model_info.return_value = mock_model_info

        mock_get = mocker.patch(
            "rzr_aikit.utils.ModelInfoFetcher.requests.get",
            return_value=self._make_response(200, hf_config),
        )
        mocker.patch(
            "rzr_aikit.utils.ModelInfoFetcher.snapshot_download",
            side_effect=Exception("not cached"),
        )

        fetcher = ModelInfoFetcher("org/standard-model", allow_internet=True)

        assert fetcher.dtype == "float16"
        assert mock_get.call_count == 1  # only config.json, no params.json

    def test_hub_source_torch_dtype_object_coerced(self, mocker):
        """Hub JSON always returns str, but coercion is exercised via _extract_dtype."""
        from rzr_aikit.utils.ModelInfoFetcher import ModelInfoFetcher

        hf_config = {"torch_dtype": "BFloat16"}  # unusual casing from some repos

        mock_hfapi = mocker.patch("rzr_aikit.utils.ModelInfoFetcher.HfApi")
        mock_model_info = MagicMock()
        mock_model_info.siblings = []
        mock_hfapi.return_value.model_info.return_value = mock_model_info

        mocker.patch(
            "rzr_aikit.utils.ModelInfoFetcher.requests.get",
            return_value=self._make_response(200, hf_config),
        )
        mocker.patch(
            "rzr_aikit.utils.ModelInfoFetcher.snapshot_download",
            side_effect=Exception("not cached"),
        )

        fetcher = ModelInfoFetcher("org/model", allow_internet=True)
        assert fetcher.dtype == "bfloat16"


class TestModelInfoFetcherMistralLocal:
    """Tests for local/cache Mistral-format repos."""

    VOXTRAL_PARAMS = {
        "dim": 4096,
        "n_layers": 32,
        "n_heads": 32,
        "vocab_size": 32768,
    }

    def test_local_source_falls_back_to_params_json(self, tmp_path, mocker):
        """Local repo without config.json but with params.json loads via normalizer."""
        from rzr_aikit.utils.ModelInfoFetcher import ModelInfoFetcher

        model_dir = tmp_path / "voxtral"
        model_dir.mkdir()
        (model_dir / "params.json").write_text(json.dumps(self.VOXTRAL_PARAMS))
        (model_dir / "consolidated.safetensors").write_bytes(b"\x00" * 100)

        mocker.patch(
            "rzr_aikit.utils.ModelInfoFetcher.AutoConfig.from_pretrained",
            side_effect=OSError("no config.json"),
        )

        fetcher = ModelInfoFetcher(str(model_dir), allow_internet=False)

        assert fetcher.dtype == "bfloat16"
        assert fetcher.quant_config == {}
        assert fetcher.total_bytes == 100
        assert fetcher.config["dim"] == 4096

    def test_local_source_raises_when_no_config_and_no_params(self, tmp_path, mocker):
        """Local repo with no config.json and no params.json re-raises OSError."""
        from rzr_aikit.utils.ModelInfoFetcher import ModelInfoFetcher

        model_dir = tmp_path / "empty"
        model_dir.mkdir()

        mocker.patch(
            "rzr_aikit.utils.ModelInfoFetcher.AutoConfig.from_pretrained",
            side_effect=OSError("no config.json"),
        )

        with pytest.raises(OSError):
            ModelInfoFetcher(str(model_dir), allow_internet=False)

    def test_local_source_torch_dtype_object_coerced(self, tmp_path, mocker):
        """New transformers: AutoConfig.to_dict() emitting torch.dtype object is coerced."""
        from rzr_aikit.utils.ModelInfoFetcher import ModelInfoFetcher

        class FakeDtype:
            def __str__(self):
                return "torch.bfloat16"

        # Patch PretrainedConfig to a sentinel class so isinstance returns False,
        # then return a plain dict from AutoConfig — the else branch assigns it directly.
        mocker.patch("rzr_aikit.utils.ModelInfoFetcher.PretrainedConfig", type(None))
        mocker.patch(
            "rzr_aikit.utils.ModelInfoFetcher.AutoConfig.from_pretrained",
            return_value={"torch_dtype": FakeDtype(), "model_type": "llama"},
        )

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "model.safetensors").write_bytes(b"\x00" * 50)

        fetcher = ModelInfoFetcher(str(model_dir), allow_internet=False)

        assert fetcher.dtype == "bfloat16"

    def test_local_source_standard_hf_config(self, tmp_path, mocker):
        """Standard HF config.json with string torch_dtype loads correctly."""
        from rzr_aikit.utils.ModelInfoFetcher import ModelInfoFetcher

        mocker.patch("rzr_aikit.utils.ModelInfoFetcher.PretrainedConfig", type(None))
        mocker.patch(
            "rzr_aikit.utils.ModelInfoFetcher.AutoConfig.from_pretrained",
            return_value={
                "torch_dtype": "float16",
                "quantization_config": {"quant_method": "awq"},
                "max_position_embeddings": 4096,
            },
        )

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "model.safetensors").write_bytes(b"\x00" * 200)

        fetcher = ModelInfoFetcher(str(model_dir), allow_internet=False)

        assert fetcher.dtype == "float16"
        assert fetcher.quant_config == {"quant_method": "awq"}
        assert fetcher.total_bytes == 200

