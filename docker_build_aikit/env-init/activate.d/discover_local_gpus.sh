#!/bin/bash

# Check local GPUs 3 times in background
nohup gpu-discovery --max-local-discovery 3 --local-gpu-only --verbose > ~/.cache/local_gpus_discovery_log.txt 2>&1 &
