# cli.py
import argparse
import json
import torch
from gpu_discovery import discover_gpus, refresh_nvml_info

def main():
    parser = argparse.ArgumentParser(
        description="Discover GPUs and benchmark FP8/FP4 support"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print detailed logs during discovery"
    )
    parser.add_argument("--no-cache", action="store_true",
        help="Ignore cached results and force fresh discovery")
    parser.add_argument(
        "--max-local-discovery", type=int, default=1,
        help="Number of attempts for local GPU discovery (default: 1)"
    )
    parser.add_argument(
        "--local-gpu-only", action="store_true",
        help="Only discover local GPUs, skip Ray cluster discovery"
    )

    args = parser.parse_args()

    verbose = False
    if args.verbose:
        verbose = True

    no_cache = False
    if args.no_cache:
        no_cache = True

    max_local_discovery = 1
    if args.max_local_discovery > 1:
        max_local_discovery = args.max_local_discovery

    local_gpu_only = False
    if args.local_gpu_only:
        local_gpu_only = True

    results = discover_gpus(verbose, no_cache, max_local_discovery, local_gpu_only)
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
