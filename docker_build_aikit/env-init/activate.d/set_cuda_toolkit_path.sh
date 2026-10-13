#!/bin/bash

# CUDA Toolkit PATH and libray PATH
export PATH="$CUDA_HOME/bin:$CUDA_HOME/extras/demo_suite:$PATH"
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=$CUDA_HOME/lib64/stubs:$LIBRARY_PATH
