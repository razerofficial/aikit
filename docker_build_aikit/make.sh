#!/bin/bash

set -e  # exit on error (-e)

# Initialize git submodules recursively
git submodule update --init --recursive
echo "✓ Git submodules initialized successfully"

print_logs() {
    echo "==================== TEST BUILD LOG ===================="
    cat docker_build_aikit/log_test.txt
    echo "==================== MAIN BUILD LOG ===================="
    cat docker_build_aikit/log.txt
}

trap print_logs ERR EXIT

: > docker_build_aikit/log_test.txt
: > docker_build_aikit/log.txt

if [ -n "$CI" ]; then
    IMAGE_NAME="$CI_REGISTRY_IMAGE/aikit:$CI_COMMIT_SHORT_SHA"
else
    IMAGE_NAME=razer:aikit
fi

PLATFORM="${2:-linux/amd64,linux/arm64}"

# docker/buildx#1170
docker run --privileged --rm tonistiigi/binfmt --install all
docker run --rm --privileged multiarch/qemu-user-static --reset -p yes -c yes

# gitlab CI
if [ -n "$CI" ]; then
    if [ -n "$1" ]; then
        # Test build
        docker buildx build \
        --target "$1" \
        --platform "$PLATFORM" \
        -f docker_build_aikit/Dockerfile_aikit .
    
    else
        # CI build: use official vllm image and push final aikit image
        docker buildx build \
        --target rzr-aikit \
        --platform "$PLATFORM" \
        --tag "$IMAGE_NAME" \
        --provenance=true --sbom=true --push \
        -f docker_build_aikit/Dockerfile_aikit .
        docker buildx imagetools inspect "$IMAGE_NAME" --format '{{ if .SBOM.SPDX }}{{ json .SBOM.SPDX }}{{ else }}{{ $found := false }}{{ range .SBOM }}{{ if not $found }}{{ json .SPDX }}{{ $found = true }}{{ end }}{{ end }}{{ end }}' > sbom.spdx.json
    fi
# local build
else
    if [ -n "$1" ]; then
        docker buildx build \
        --target "$1" \
        --platform "$PLATFORM" \
        -f docker_build_aikit/Dockerfile_aikit . \
        1> docker_build_aikit/log_test.txt 2>&1
    else
        docker buildx build \
        --target rzr-aikit \
        --platform "$PLATFORM" \
        --tag "$IMAGE_NAME" \
        -f docker_build_aikit/Dockerfile_aikit . \
        1> docker_build_aikit/log.txt 2>&1
    fi
fi
