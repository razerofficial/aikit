import os
import re
import json
import requests

from typing import Optional
from transformers import AutoConfig, PretrainedConfig
from huggingface_hub import snapshot_download, HfApi, ModelInfo, hf_hub_url
from huggingface_hub.errors import LocalEntryNotFoundError


def _normalize_mistral_params(raw: dict) -> dict:
    """Shape a Mistral params.json into a HF-config-compatible dict."""
    quant = raw.get("quantization")
    return {
        **raw,
        "torch_dtype": raw.get("dtype", "bfloat16"),
        "quantization_config": quant if isinstance(quant, dict) else {},
    }


def _extract_dtype(config: dict) -> str:
    """Return a lowercase dtype string from a config dict.

    Handles both old transformers (torch_dtype as str) and new transformers
    (dtype as torch.dtype object), as well as raw JSON configs (always str).
    Falls back to 'float32' when absent.
    """
    value = config.get("torch_dtype") or config.get("dtype")
    if value is None:
        return "float32"
    if isinstance(value, str):
        return value.lower()
    # torch.dtype object e.g. torch.bfloat16 -> "bfloat16"
    return str(value).rsplit(".", 1)[-1].lower()


class ModelInfoFetcher:
    def __init__(
        self,
        model: str,
        access_token: Optional[str] = None,
        allow_internet: bool = True,
    ):
        self.model = model
        self.access_token = access_token
        self.allow_internet = allow_internet
        self.is_local = os.path.isdir(model)

        if self.is_local:
            self.source = "local"
            self.model_path = model
        elif self._try_find_in_cache(model):
            self.source = "cache"
        elif allow_internet:
            self.source = "hub"
        else:
            raise ValueError(
                f"Model '{model}' not found locally and internet access disabled."
            )

        self.config = None
        self.dtype = None
        self.quant_config = None
        self.weight_files = None
        self.total_bytes = None

        self._load_model_info()

    def _try_find_in_cache(self, model_id: str) -> bool:
        try:
            local_dir = snapshot_download(repo_id=model_id, local_files_only=True)
            self.model_path = local_dir
            return True
        except LocalEntryNotFoundError as e:
            pass
        except Exception as e:
            pass

        return False

    def _load_model_info(self):
        if self.source in ("local", "cache"):
            try:
                config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
                self.config = config.to_dict() if isinstance(config, PretrainedConfig) else config
            except (OSError, EnvironmentError, ValueError):
                params_path = os.path.join(self.model_path, "params.json")
                if not os.path.exists(params_path):
                    raise
                with open(params_path) as f:
                    self.config = _normalize_mistral_params(json.load(f))
            self.dtype = _extract_dtype(self.config)
            self.quant_config = self.config.get("quantization_config", {})
            self.weight_files = self._get_local_weight_files(self.model_path)
            grouped_weights = {
                    ext: sum(size for filename, size in files)
                    for ext, files in self.weight_files.items()
                }
            priority = ["safetensors", "bin", "gguf", "ggml"]
            self.total_bytes = next((grouped_weights[ext] for ext in priority if ext in grouped_weights), None)
            if self.allow_internet:
                self.hfapi = HfApi()
                self._model_info: ModelInfo = self.hfapi.model_info(self.model, files_metadata=True)
        elif self.source == "hub":
            try:
                # get weight size
                self.hfapi = HfApi()
                self._model_info: ModelInfo = self.hfapi.model_info(self.model, files_metadata=True)
                self.weight_files = self._get_remote_weight_files()
                grouped_weights = {
                    ext: sum(size for filename, size in files if "/" not in filename)
                    for ext, files in self.weight_files.items()
                }
                priority = ["safetensors", "bin", "gguf", "ggml"]
                self.total_bytes = next((grouped_weights[ext] for ext in priority if ext in grouped_weights), None)
                # get dtype and quantization_config
                self.config = self._get_remote_config_json()
                self.dtype = _extract_dtype(self.config)
                self.quant_config = self.config.get("quantization_config", {})

            except requests.exceptions.HTTPError as e:
                if (e.response.status_code == 401) and ("unauthorized" in e.response.reason.lower()):
                    from rzr_aikit import HuggingfaceAccessTokenRequired
                    raise HuggingfaceAccessTokenRequired("Huggingface Access Token is required") from e

                print(f"_load_model_info() exception {e.__class__} : {e}")
                pass
            except FileNotFoundError:
                raise
            except Exception as e:
                print(f"_load_model_info() exception {e.__class__} : {e}")
                pass

        else:
            raise RuntimeError("Unknown model source.")

    def _get_remote_weight_files(self):
        weight_files: dict[str, list[tuple[str, int]]] = {}
        for f in self._model_info.siblings:
            if f.rfilename.endswith((".bin", ".safetensors", ".gguf", ".ggml")):
                ext = f.rfilename.split(".")[-1]
                if ext not in weight_files:
                    weight_files[ext] = []
                weight_files[ext].append((f.rfilename, f.size))
        return weight_files

    def _get_remote_config_json(self):
        headers = {"Authorization": f"Bearer {self.access_token}"} if self.access_token else {}

        for filename in ("config.json", "params.json"):
            url = hf_hub_url(repo_id=self.model, filename=filename, repo_type="model")
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 404:
                continue
            response.raise_for_status()
            data = response.json()
            if filename == "params.json":
                data = _normalize_mistral_params(data)
            return data

        raise FileNotFoundError(
            f"Neither config.json nor params.json found in {self.model}@main"
        )

    @staticmethod
    def _get_local_weight_files(local_dir: str):
        weight_files: dict[str, list[tuple[str, int]]] = {}
        for file in os.scandir(local_dir):
            if file.is_file() and file.name.endswith((".bin", ".safetensors", ".gguf", ".ggml")):
                ext = file.name.split(".")[-1]
                if ext not in weight_files:
                    weight_files[ext] = []
                weight_files[ext].append((file.name, file.stat().st_size))
        return weight_files

    @staticmethod
    def get_repo_id_from_model_path(path: str) -> str | None:
        """
        Extracts the Hugging Face repo ID in the format 'org/model' from a local cache path.

        Args:
            path: Full local model path, e.g.
                  ~/.cache/huggingface/hub/models--org--model/snapshots/...

        Returns:
            A string like 'org/model', or None if the pattern doesn't match.
        """
        match = re.search(r"models--(.+?)--([^/]+)", path)
        if match:
            return f"{match.group(1)}/{match.group(2)}"
        return None

    def summarize(self) -> dict:
        return {
            "model": self.model,
            "source": self.source,
            "torch_dtype": self.dtype,
            "quantization_config": self.quant_config,
            "weight_size_bytes": self.total_bytes,
            "weight_files": self.weight_files,
        }

    def pretty_print(self):
        summary = self.summarize()
        print(f"\n== {summary['model']} ({summary['source']}) ==")
        print(f"- Torch dtype         : {summary['torch_dtype']}")
        print(f"- Quantization config : {summary['quantization_config']}")

        if summary['weight_size_bytes'] is None:
            print(f"- Total weight size   : None")
        else:
            print(f"- Total weight size   : {summary['weight_size_bytes']} Bytes {summary['weight_size_bytes'] / (1024 ** 3):.2f} GB")

        if summary['weight_size_bytes'] is None:
            print("\nWeight Files: None")
        else:
            for name, size in summary["weight_files"]:
                print(f"  {name:60} {size} Bytes {size / (1024 ** 2):.2f} MB")

