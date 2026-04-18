#!/usr/bin/env python3
# run_vllm.py
import os
import re
import sys
import logging
import argparse

from vllm import EngineArgs, LLMEngine
from vllm.config import VllmConfig
from vllm.utils.argparse_utils import FlexibleArgumentParser

def get_vllm_engine_version() -> str:
    """
    Checks and returns the active vLLM engine version ('V0' or 'V1').

    Returns:
        str: The engine version ('V0' or 'V1').
    """
    use_v1_env = os.environ.get("VLLM_USE_V1", "1").lower()

    # In recent versions of vLLM, V1 is the default.
    # The environment variable being unset is equivalent to it being "1".
    if use_v1_env in ["0", "false"]:
        return "V0"
    else:
        return "V1"

def main() -> int:
    log_critical = False

    try:
        # a.  Start with a plain parser
        parser = FlexibleArgumentParser(description="vLLM offline runner")

        # b.  Add custom flag
        parser.add_argument("--razer-enable-log-critical", action="store_true", help="Enable critical logging for vLLM")

        # c.  Inject every vLLM flag in one line
        parser = EngineArgs.add_cli_args(parser)

        # d.  Parse *all* CLI flags (your own + vLLM)
        args = parser.parse_args()

        # e.  Conditionally enable logging
        if args.razer_enable_log_critical:
            log_critical = True
            vllm_logger = logging.getLogger("vllm")
            vllm_logger.setLevel(logging.CRITICAL)

        # f.  Convert the Namespace to EngineArgs dataclass
        eng_args = EngineArgs.from_cli_args(args)

        # g.  Bring up the engine.
        engine = LLMEngine.from_engine_args(eng_args)

        # h.  Ask the scheduler what context length it finally decided on.
        vllm_config = engine.vllm_config
        RAZER_MAX_MODEL_LEN = vllm_config.model_config.max_model_len
        RAZER_MAX_NUM_SEQS = vllm_config.scheduler_config.max_num_seqs
        RAZER_MAX_NUM_BATCHED_TOKENS = vllm_config.scheduler_config.max_num_batched_tokens

        # i.  Print VLLM parameters
        print(f"RAZER_MAX_MODEL_LEN={RAZER_MAX_MODEL_LEN}", file=sys.stderr)
        print(f"RAZER_MAX_NUM_SEQS={RAZER_MAX_NUM_SEQS}", file=sys.stderr)
        print(f"RAZER_MAX_NUM_BATCHED_TOKENS={RAZER_MAX_NUM_BATCHED_TOKENS}", file=sys.stderr)

        return 200

    except ValueError as e:
        if get_vllm_engine_version() == "V0":
            error_message = str(e)

            if not log_critical :
                print(f"V0 {error_message}", file=sys.stderr)

            match = re.search(r"tokens that can be stored in KV cache\s*\((\d+)\)", error_message)
            if match:
                RAZER_MAX_MODEL_LEN = int(match.group(1))
                print(f"RAZER_MAX_MODEL_LEN={RAZER_MAX_MODEL_LEN}", file=sys.stderr)

                #continue
                return 100

        return -1
    except Exception as e:
        vllm_engine_version : str = get_vllm_engine_version()

        if vllm_engine_version == "V1":
            #continue
            return 101

        elif vllm_engine_version == "V0":
            error_message = str(e)
            print(f"V0 {error_message}", file=sys.stderr)

        return -1

if __name__ == "__main__":
    sys.exit(main())

