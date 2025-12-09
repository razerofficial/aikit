#!/bin/bash

# Prevent sourcing: must be executed, not sourced
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    echo "Error: This script is meant to be executed, not sourced." >&2
    return
fi

# get absolute path
if command -v realpath >/dev/null 2>&1; then
    SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
else
    echo "Error: realpath is not installed." >&2
    exit 1
fi

# Directory containing the script
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"

# Move into the parent directory of the script's location
cd "$SCRIPT_DIR/.."

echo "Working directory set to: $(pwd)"

# update cutlass submodule
git submodule update --init --recursive

docker build --tag razer:vllm -f docker_build_vllm/Dockerfile_vllm .  1> log.txt 2>&1
