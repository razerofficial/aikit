#!/bin/bash
# Functional test runner for AIKit
# Usage: ./run_tests.sh [test-category]

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Change to functional test directory
cd "$(dirname "$0")"

print_usage() {
    echo "Usage: $0 [category]"
    echo ""
    echo "Categories:"
    echo "  all             Run all tests (default)"
    echo "  fast            Run fast tests only (exclude slow tests)"
    echo "  llm             Run LLM inference tests"
    echo "  image           Run image generation tests"
    echo "  audio           Run audio generation tests"
    echo "  voice           Run voice agent tests"
    echo "  core            Run core package tests"
    echo "  no-integration  Run all except integration tests"
    echo "  ci              Run CI-safe tests (no GPU, network, or integration)"
    echo ""
    echo "Examples:"
    echo "  $0              # Run all tests"
    echo "  $0 fast         # Run fast tests only"
    echo "  $0 llm          # Run LLM tests"
    echo "  $0 ci           # Run CI-safe tests"
}

run_tests() {
    local marker="$1"
    local description="$2"

    echo -e "${GREEN}Running $description...${NC}"
    pytest -v $marker
    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✓ $description passed${NC}"
    else
        echo -e "${RED}✗ $description failed${NC}"
        exit $exit_code
    fi
}

# Parse command line argument
CATEGORY="${1:-all}"

case "$CATEGORY" in
    all)
        run_tests "" "all functional tests"
        ;;

    fast)
        run_tests "-m 'not slow'" "fast tests"
        ;;

    llm)
        run_tests "-m llm" "LLM inference tests"
        ;;

    image)
        run_tests "-m image" "image generation tests"
        ;;

    audio)
        run_tests "-m audio" "audio generation tests"
        ;;

    voice)
        run_tests "-m voice" "voice agent tests"
        ;;

    core)
        run_tests "test_core_packages.py" "core package tests"
        ;;

    no-integration)
        run_tests "-m 'not integration'" "non-integration tests"
        ;;

    ci)
        echo -e "${YELLOW}Running CI-safe tests (no GPU, network, or integration)${NC}"
        run_tests "-m 'not slow and not requires_gpu and not requires_network and not integration'" "CI-safe tests"
        ;;

    help|-h|--help)
        print_usage
        exit 0
        ;;

    *)
        echo -e "${RED}Unknown category: $CATEGORY${NC}"
        echo ""
        print_usage
        exit 1
        ;;
esac

echo -e "${GREEN}All requested tests completed successfully!${NC}"
