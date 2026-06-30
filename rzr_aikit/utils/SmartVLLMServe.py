import argparse
import dataclasses
import json
import os
import re
import subprocess
import sys

from argparse import BooleanOptionalAction
from enum import Enum
from typing import Any, List

from vllm import EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser


class SmartVLLMServe:
    """
    Construct an LLMEngine from a model name/path plus any EngineArgs-supported flags *exactly* as you would write them on the command line.

    Parameters
    ----------
    model_or_path : str
        A Hugging Face model ID (e.g. "meta-llama/Llama3.2-3B") or a local path.
    *cli_flags : str
        Any sequence of CLI-style flags supported by `vllm.EngineArgs`.
        Example: "--enable-eager", "--max-model-len", "4096".
    **kw_overrides
        Optional keyword overrides (handy when a field lacks a CLI flag or when you prefer pure Python).
        Each keyword must match an EngineArgs attribute name.

    Raises
    ------
    ValueError - when an unknown flag is supplied.
    """

    def __init__(self, model_or_path: str, *cli_flags: str, **kw_overrides: Any):
        # ---- 1.  Parse CLI-style flags -----------------------------------------
        # Pretend the first positional argument is --model
        cli_argv: List[str] = ["--model", model_or_path, *cli_flags]

        parser = FlexibleArgumentParser(
            prog="vllm_wrapper",
            add_help=False,  # we don't expose -h from here
            allow_abbrev=False,
        )
        parser = EngineArgs.add_cli_args(parser)

        # ---- 2.  Build EngineArgs metadata -------------------------------------
        self.field_metadata = {}
        for action in parser._actions:
            # The 'dest' is the name of the attribute in the dataclass
            if action.dest and action.option_strings:
                # We map the field name (dest) to the first (and usually primary)
                # CLI flag, e.g., 'model' -> '--model'
                self.field_metadata[action.dest] = {
                    "option_strings": action.option_strings,
                    "action": type(action).__name__,
                }

        # ---- 3.  Build default EngineArgs --------------------------------------
        default_parsed = parser.parse_args(["--model", "Raer"])
        self.default_engine_args: EngineArgs = EngineArgs.from_cli_args(default_parsed)

        # ---- 4.  Build EngineArgs object ---------------------------------------
        # parse_known_intermixed_args keeps argument order (Python >= 3.9).
        parsed, extra = parser.parse_known_intermixed_args(cli_argv)
        if extra:
            raise ValueError(f"Unrecognized EngineArgs flag(s): {extra}")

        self.engine_args: EngineArgs = EngineArgs.from_cli_args(parsed)

        # ---- 5.  Apply any programmatic keyword overrides *after* CLI parsing --
        for key, value in kw_overrides.items():
            if not hasattr(self.engine_args, key):
                raise ValueError(f"{key!r} is not a valid EngineArgs field")
            setattr(self.engine_args, key, value)

    def get_cli_args(self) -> List[str]:
        """
        Generates a CLI argument list for fields that differ between two EngineArgs instances.
        Handles enums, dictionaries, lists, and boolean flags with negative options.

        Returns:
            List of CLI arguments (e.g., ['--model', 'new_model', '--no-flag'])
        """

        changed_args = []

        for field in dataclasses.fields(self.default_engine_args):
            old_val = getattr(self.default_engine_args, field.name)
            new_val = getattr(self.engine_args, field.name)

            # Skip unchanged fields
            if old_val == new_val:
                continue

            # Get all option strings (support both singular and plural metadata keys)
            metadata = self.field_metadata.get(field.name)
            option_strings = metadata.get("option_strings")
            action = metadata.get("action")

            # Skip if no CLI flags
            if not option_strings:
                continue

            # Handle BooleanOptionalAction (--flag/--no-flag)
            if action == "BooleanOptionalAction":
                # Use positive flag for True, negative flag for False
                if len(option_strings) >= 2:
                    flag = option_strings[0] if new_val else option_strings[1]
                    changed_args.append(flag)
                elif option_strings:
                    # Fallback: use single flag with value representation
                    changed_args.append(option_strings[0])
                    changed_args.append(str(new_val))

            # Handle standard boolean flags
            elif action in ("_StoreTrueAction", "_StoreFalseAction"):
                if (action == "_StoreTrueAction" and new_val) or (
                    action == "_StoreFalseAction" and not new_val
                ):
                    if option_strings:
                        changed_args.append(option_strings[0])

            # Handle non-boolean fields
            else:
                if not option_strings:
                    continue
                flag = option_strings[0]

                # Handle lists (including lists of enums)
                if isinstance(new_val, list):
                    processed_items = [
                        item.value if isinstance(item, Enum) else item
                        for item in new_val
                    ]
                    changed_args.append(flag)
                    changed_args.append(",".join(map(str, processed_items)))

                # Handle enum values
                elif isinstance(new_val, Enum):
                    changed_args.append(flag)
                    changed_args.append(str(new_val.value))

                # Handle dictionaries
                elif isinstance(new_val, dict):
                    changed_args.append(flag)
                    changed_args.append(json.dumps(new_val))

                # Handle all other types
                else:
                    # Some nested/structured vLLM args (e.g. rope_parameters)
                    # may be represented as non-serializable helper objects
                    # (like ArgsKwargs). Passing their string repr to CLI
                    # breaks pydantic validation. Only include if JSON-serializable.
                    try:
                        json.dumps(new_val)
                        changed_args.append(flag)
                        changed_args.append(str(new_val))
                    except (TypeError, ValueError):
                        # Special-case: skip rope_parameters unless provided as dict
                        # so vLLM will rely on its internal defaults.
                        if field.name == "rope_parameters":
                            continue
                        # Otherwise, conservatively skip unrecognized complex types.
                        continue

        return changed_args

    def get_cli_string(self) -> str:
        """
        Export a single-line CLI string for shell/bash use, reflecting all resolved flags.
        Example output:
            --model 'meta-llama/Llama-3.2-3B' --enable-eager --max-model-len 4096

        Returns
        -------
        str : shell-quoted command-line string
        """
        import shlex

        args = self.get_cli_args()
        return " ".join(shlex.quote(arg) for arg in args)

    def get_max_model_len(self) -> int:
        # 1. Get CLI arguments
        cli_args = self.get_cli_args()
        command = ["run_vllm"] + cli_args

        # 2. Run the subprocess, capturing stderr to print it conditionally
        proc = subprocess.run(
            command,
            capture_output=True,  # Captures both stdout and stderr
            text=True,  # Decodes output as text
        )

        if proc.returncode == 255:
            print(proc.returncode)
            print(proc.stderr)

        # 3. Handle specific return codes
        if proc.returncode == 100:
            # Define the regex to find the value of RAZER_MAX_MODEL_LEN
            pattern = r"RAZER_MAX_MODEL_LEN=(\d+)"

            # Search for the pattern in the message
            match = re.search(pattern, proc.stderr)

            if match:
                RAZER_MAX_MODEL_LEN = int(match.group(1))
                self.engine_args.max_model_len = RAZER_MAX_MODEL_LEN
                print(f"RAZER_MAX_MODEL_LEN={RAZER_MAX_MODEL_LEN}", file=sys.stderr)
            else:
                print("Could not find the RAZER_MAX_MODEL_LEN value.")

        elif proc.returncode == 101:
            # Define the specific regular expression
            pattern = r"the estimated maximum model length is (\d+)"

            # Search for the pattern in the message
            match = re.search(pattern, proc.stderr)
            if match:
                RAZER_MAX_MODEL_LEN = int(match.group(1))
                self.engine_args.max_model_len = RAZER_MAX_MODEL_LEN
                print(f"RAZER_MAX_MODEL_LEN={RAZER_MAX_MODEL_LEN}", file=sys.stderr)
                return proc.returncode
            else:
                print(proc.returncode)
                print(proc.stderr)
                return -1

        elif proc.returncode == 200:
            # Define the regex to find the value of RAZER_MAX_MODEL_LEN
            pattern = r"RAZER_MAX_MODEL_LEN=(\d+)"

            # Search for the pattern in the message
            match = re.search(pattern, proc.stderr)

            if match:
                RAZER_MAX_MODEL_LEN = int(match.group(1))
                self.engine_args.max_model_len = RAZER_MAX_MODEL_LEN
                print(f"RAZER_MAX_MODEL_LEN={RAZER_MAX_MODEL_LEN}", file=sys.stderr)
            else:
                print("Could not find the RAZER_MAX_MODEL_LEN value.")

            # Define the regex to find the value of RAZER_MAX_NUM_SEQS
            pattern = r"RAZER_MAX_NUM_SEQS=(\d+)"

            # Search for the pattern in the message
            match = re.search(pattern, proc.stderr)

            if match:
                RAZER_MAX_NUM_SEQS = int(match.group(1))
                self.engine_args.max_num_seqs = RAZER_MAX_NUM_SEQS
                print(f"RAZER_MAX_NUM_SEQS={RAZER_MAX_NUM_SEQS}")
            else:
                print("Could not find the RAZER_MAX_NUM_SEQS value.")

            # Define the regex to find the value of RAZER_MAX_NUM_BATCHED_TOKENS
            pattern = r"RAZER_MAX_NUM_BATCHED_TOKENS=(\d+)"

            # Search for the pattern in the message
            match = re.search(pattern, proc.stderr)

            if match:
                RAZER_MAX_NUM_BATCHED_TOKENS = int(match.group(1))
                self.engine_args.max_num_batched_tokens = RAZER_MAX_NUM_BATCHED_TOKENS
                print(f"RAZER_MAX_NUM_BATCHED_TOKENS={RAZER_MAX_NUM_BATCHED_TOKENS}")
            else:
                print("Could not find the RAZER_MAX_NUM_BATCHED_TOKENS value.")

        # 4. Return the exit code
        return proc.returncode

    def determine_dtype(self) -> None:
        if self.engine_args.dtype == "auto":
            # get model config
            model_config = None
            quant_config = None
            dtype = None

            try:
                from rzr_aikit.utils.ModelInfoFetcher import ModelInfoFetcher
                from rzr_aikit import HuggingfaceAccessTokenRequired

                fetcher = ModelInfoFetcher(self.engine_args.model)
                model_config = fetcher.config
                quant_config = fetcher.quant_config
                dtype = fetcher.dtype
            except HuggingfaceAccessTokenRequired as e:
                from rzr_aikit.utils.HuggingfaceHubToken import HuggingfaceHubToken

                try:
                    hug = HuggingfaceHubToken()
                    token = hug.get_access_token()

                    fetcher = ModelInfoFetcher(self.engine_args.model, token)
                    model_config = fetcher.config
                    quant_config = fetcher.quant_config
                    dtype = fetcher.dtype
                except HuggingfaceAccessTokenRequired as e:
                    raise

            # skip for quantized model
            if quant_config:
                return

            # check whether dtype is bfloat16
            if dtype == "bfloat16":
                import torch

                num_devices = torch.cuda.device_count()
                devices_required = (
                    self.engine_args.tensor_parallel_size
                    * self.engine_args.pipeline_parallel_size
                )

                # local GPUs
                if devices_required <= num_devices:
                    # check bfloat16 suports
                    all_gpus_support_bfloat16 = True

                    for device_id in range(devices_required):
                        device_name = torch.cuda.get_device_name(device_id)
                        capability = torch.cuda.get_device_capability(device_id)

                        # Native bfloat16 requires compute capability >= 8.0 (Ampere+)
                        if capability[0] < 8:
                            all_gpus_support_bfloat16 = False
                            break

                    if not all_gpus_support_bfloat16:
                        self.engine_args.dtype = "half"

    def serve(
        self,
        host: str | None = None,
        port: int | None = None,
        *,
        detach: bool = True,
        extra: list[str] | None = None,
        env: dict | None = None,
    ) -> subprocess.Popen | subprocess.CompletedProcess:
        """
        Launch `vllm serve` with resolved CLI flags.
        """
        args = self.get_cli_args()

        cmd = ["vllm", "serve", *args[1:]]

        if host:
            cmd += ["--host", host]
        if port:
            cmd += ["--port", str(port)]
        if extra:
            cmd += extra

        final_env = os.environ.copy()
        if env:
            final_env.update(env)

        print(cmd)

        if detach:
            log_file = open("/tmp/rzr_aikit.log", "w")
            return subprocess.Popen(cmd, env=final_env, stdout=log_file, stderr=subprocess.STDOUT, encoding='utf-8')

        return subprocess.run(cmd, check=True, env=final_env)
