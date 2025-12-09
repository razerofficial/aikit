#!/bin/bash

set -e  # exit on error (-e)

print_logs() {
    echo "==================== TEST BUILD LOG ===================="
    cat docker_build_aikit/log_test.txt
    echo "==================== MAIN BUILD LOG ===================="
    cat docker_build_aikit/log.txt
}

trap print_logs ERR EXIT

: > docker_build_aikit/log_test.txt
: > docker_build_aikit/log.txt

VLLM_IMAGE="$CI_REGISTRY_IMAGE/vllm:latest"

if [ -n "$CI" ]; then
    IMAGE_NAME="$CI_REGISTRY_IMAGE/aikit:$CI_COMMIT_SHORT_SHA"
else
    IMAGE_NAME=aikit
fi

# gitlab CI
if [ -n "$CI" ]; then
    if [ -n "$1" ]; then
        # Test build: pull vllm from registry
        docker build \
        --target "$1" \
        --build-arg VLLM_IMAGE="$VLLM_IMAGE" \
        -f docker_build_aikit/Dockerfile_aikit . \
    
    else
        # CI build: pull vllm from registry and push final aikit image
        docker buildx build \
        --target rzr-aikit \
        --build-arg VLLM_IMAGE="$VLLM_IMAGE" \
        --tag "$IMAGE_NAME" \
        --provenance=true --sbom=true --push \
        -f docker_build_aikit/Dockerfile_aikit .
        docker buildx imagetools inspect "$IMAGE_NAME" --format "{{ json .SBOM.SPDX }}" > sbom.spdx.json
    fi
# local build
else
    if [ -n "$1" ]; then
        docker build \
        --target "$1" \
        -f docker_build_aikit/Dockerfile_aikit . \
        1> docker_build_aikit/log_test.txt 2>&1
    else
        docker build \
        --target rzr-aikit \
        --tag "$IMAGE_NAME" \
        -f docker_build_aikit/Dockerfile_aikit . \
        1> docker_build_aikit/log.txt 2>&1
    fi
fi
