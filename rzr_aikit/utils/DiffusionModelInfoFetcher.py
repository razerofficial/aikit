import os
import re
import json
import requests
import struct

from typing import Optional
from huggingface_hub import snapshot_download, HfApi, ModelInfo, hf_hub_url
from huggingface_hub.errors import LocalEntryNotFoundError

from vllm_omni.diffusion.registry import _DIFFUSION_MODELS


class DiffusionModelInfoFetcher:
    """
    Fetcher for diffusion models based on ModelInfoFetcher.
    
    Focuses on retrieving:
    - model_info (tag, downloads, license, update time)
    - index (model_index.json)
    - weight_files (files based on architecture)
    """
    
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

        self.model_info = None
        self.weight_files = None
        self._model_architecture = None

        self._load_model_info()

    def _try_find_in_cache(self, model_id: str) -> bool:
        try:
            local_dir = snapshot_download(repo_id=model_id, local_files_only=True)
            self.model_path = local_dir
            return True
        except LocalEntryNotFoundError:
            pass
        except Exception:
            pass

        return False

    def _load_model_info(self):
        """Load model info based on source."""
        if self.source in ("local", "cache"):
            self.index = self._get_local_index()
            self._model_architecture = self.index.get("_class_name")
            if self._model_architecture not in _DIFFUSION_MODELS:
                raise ValueError(f"Unsupported architecture: {self._model_architecture}")
            
            self.weight_files = self._get_local_weight_files(self.model_path, self.index)
            grouped_weights = {}
            for ext, files in self.weight_files.items():
                total_size = 0
                for filename, size in files:
                    total_size += size * (16 / self._get_local_dtype(filename))
                grouped_weights[ext] = total_size
            priority = ["safetensors", "bin"]
            self.total_bytes = next((grouped_weights[ext] for ext in priority if ext in grouped_weights), 0)
            
            if self.allow_internet:
                self.hfapi = HfApi()
                self.model_info: ModelInfo = self.hfapi.model_info(self.model, files_metadata=True)

        elif self.source == "hub":
            try:
                self.hfapi = HfApi()
                self.model_info: ModelInfo = self.hfapi.model_info(self.model, files_metadata=True)
                
                self.index = self._get_remote_index_json()
                self._model_architecture = self.index.get("_class_name")
                if self._model_architecture not in _DIFFUSION_MODELS:
                    raise ValueError(f"Unsupported architecture: {self._model_architecture}")
                
                self.weight_files = self._get_remote_weight_files(self.index)
                grouped_weights = {}
                for ext, files in self.weight_files.items():
                    total_size = 0
                    for filename, size in files:
                        total_size += size * (16 / self._get_remote_dtype(filename))
                    grouped_weights[ext] = total_size
                priority = ["safetensors", "bin"]
                self.total_bytes = next((grouped_weights[ext] for ext in priority if ext in grouped_weights), 0)

            except requests.exceptions.HTTPError as e:
                if (e.response.status_code == 401) and ("unauthorized" in e.response.reason.lower()):
                    from rzr_aikit import HuggingfaceAccessTokenRequired
                    raise HuggingfaceAccessTokenRequired("Huggingface Access Token is required") from e
                print(f"_load_model_info() exception {e.__class__} : {e}")
                pass
            except Exception as e:
                print(f"_load_model_info() exception {e.__class__} : {e}")
                pass
        else:
            raise RuntimeError("Unknown model source.")

    def _get_local_index(self):
        """Load index from local model directory."""
        index_path = os.path.join(self.model_path, "model_index.json")
        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _get_local_weight_files(self, local_dir: str, index: dict):
        """Get weight files from local directory."""
        weight_files: dict[str, list[tuple[str, int]]] = {}
        subfolders = [key for key, value in index.items() if isinstance(value, list)]
        
        for subfolder in subfolders:
            subfolder_path = os.path.join(local_dir, subfolder)
            if not os.path.isdir(subfolder_path):
                continue
                
            for file in os.scandir(subfolder_path):
                if file.is_file() and file.name.endswith((".bin", ".safetensors")):
                    ext = file.name.split(".")[-1]
                    if ext not in weight_files:
                        weight_files[ext] = []
                    weight_files[ext].append((f"{subfolder_path}/{file.name}", file.stat().st_size))
        return weight_files
    
    def _get_local_dtype(self, path: str):
        """Extract dtype from local safetensors file."""
        with open(path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))

        for name, meta in header.items():
            if name == "__metadata__":
                continue
            return int(''.join(filter(str.isdigit, meta["dtype"])))
        return None

    def _get_remote_index_json(self):
        """Download model_index.json from HuggingFace Hub."""
        # Generate the correct URL for model_index.json
        url = hf_hub_url(repo_id=self.model, filename="model_index.json", repo_type="model")

        headers = {}
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"

        # Download directly to memory
        response = requests.get(url, headers=headers, timeout=10)

        # Handle possible errors
        if response.status_code == 404:
            raise FileNotFoundError(f"model_index.json not found in {self.model}@main")
        response.raise_for_status()

        return response.json()
    
    def _get_remote_weight_files(self, index: dict):
        """Get weight files from HuggingFace Hub."""
        weight_files: dict[str, list[tuple[str, int]]] = {}
        subfolders = [key for key, value in index.items() if isinstance(value, list)]

        for subfolder in subfolders:
            for file in self.model_info.siblings:
                if file.rfilename.endswith((".bin", ".safetensors")) and file.rfilename.startswith(f"{subfolder}/"):
                    ext = file.rfilename.split(".")[-1]
                    if ext not in weight_files:
                        weight_files[ext] = []
                    weight_files[ext].append((file.rfilename, file.size))
        return weight_files
    
    def _get_remote_dtype(self, filename: str):
        """Extract dtype from remote safetensors file."""
        url =  hf_hub_url(self.model, filename=filename)
        headers = {}
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        headers['Range'] = 'bytes=0-7'
        response = requests.get(url, headers=headers)
        length_of_header = struct.unpack('<Q', response.content)[0]
        headers['Range'] = f'bytes=8-{7 + length_of_header}'
        response = requests.get(url, headers=headers)
        header = response.json()

        for name, meta in header.items():
            if name == "__metadata__":
                continue
            return int(''.join(filter(str.isdigit, meta["dtype"])))
        return None

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
