# Known Issues and Solutions

This document contains common issues you might encounter when using Razer AIKit and their corresponding solutions.

## Memory and Performance Issues

### 1. Windows with VRAM Less Than 10GB

**Problem**: Out of memory errors or poor performance on Windows systems with limited VRAM.

**Solution**: Reduce GPU memory utilization to prevent memory overflow:

```bash
rzr-aikit model run <your-model> --gpu-memory-utilization 0.8
```

**Note**: You can adjust the value (0.8 = 80%) based on your available VRAM. Lower values use less memory but may impact performance.

---

## Platform-Specific Issues

### 2. Windows Inference Not Working

**Problem**: The AIKit model runs successfully, but inference commands like `rzr-aikit model generate` fail or port 8000 is not accessible for inferencing.

**Symptoms**:
- Model starts and loads without errors
- `rzr-aikit model generate` commands fail or hang
- Cannot connect to port 8000 for inference requests

**Solution**: After the model run completes but before running generate commands, run the following command to trigger the network connection:

```bash
python -m http.server --bind 0.0.0.0 8000
```

**Expected behavior**: The command will return an error indicating that port 8000 is already in use by vLLM (this is normal). After running this command, the connection to vLLM on port 8000 should become accessible.

**Additional Notes**: 
- This is typically needed when running in WSL2 on Windows
- Run this command after `rzr-aikit model run` completes successfully
- The error about port 8000 being occupied is expected and indicates vLLM is running
- After the error appears, you can proceed with `rzr-aikit model generate` commands
- Ensure Windows Firewall allows the connection if needed

### 3. GB10 (DGX Spark) Related Issues

**Problem**: vLLM v0.28.0 running on SM 121 (GB10 DGX Spark) hardware experiences crashes, deadlocks, and kernel failures due to incorrect hardware capability detection and unsupported optimizations.

**Symptoms**:
- Runtime crashes during matrix multiplication operations
- Inference pipeline hangs indefinitely with no response
- Kernel launch failures on models using Multi-Latent Attention (MLA)
- vLLM server starts but fails to complete inference requests

**Solution**: Set few environment variables when starting the container

```bash
docker run -it \
  --restart=unless-stopped \
  --gpus all \
  --ipc host \
  --network host \
  --mount type=bind,source=$HOME/.cache/huggingface,target=/var/aikit/.cache/huggingface \
  --env HUGGING_FACE_HUB_TOKEN=<YOUR_TOKEN> \
  --env VLLM_SKIP_DEEP_GEMM=1 \
  --env VLLM_DISABLE_PLE=1 \
  --env VLLM_FORCE_STANDARD_ATTENTION=1 \
  razerofficial/aikit:latest
```

**What each workaround fixes**:

1. VLLM_SKIP_DEEP_GEMM=1
    - Issue: DeepGEMM false support detection
    - Details: vLLM incorrectly detects SM 121 supports DeepGEMM operations, but execution crashes
    - Fallback: Uses standard GEMM implementations instead
2. VLLM_DISABLE_PLE=1
    - Issue: Pipeline Lookahead Encoding (PLE) offload deadlocks
    - Details: At TP=1 (single GPU), PLE prefetch causes indefinite hangs
    - Fallback: Disables PLE prefetching optimization
3. VLLM_FORCE_STANDARD_ATTENTION=1
    - Issue: Multi-Latent Attention (MLA) decode shared memory overflow
    - Details: MLA kernels allocate more shared memory than SM 121 supports
    - Fallback: Forces standard attention (FlashAttention, xFormers) instead of MLA kernels

**Expected behavior**:
- vLLM server starts and serves requests successfully
- Inference completes without crashes or hangs

**Additional Notes**:
- These workarounds are required for SM 121 (GB10 DGX Spark) hardware only
- SM 120 (Jetson Orin, RTX 50 series) may not need all three workarounds
- Performance impact is minimal (~0-5% slower) for most workloads
- These are temporary workarounds until vLLM adds proper SM 121 support in future releases
- Official vLLM ARM64 images support SM 121 via PTX compatibility when these workarounds are applied

**Build-time workaround**: `--build-arg BUILD_BASE_IMAGE=pytorch/manylinuxaarch64-builder:cuda13.0` and `--build-arg torch_cuda_arch_list="12.0 12.0a 12.1 12.1a+PTX"` to compile the missing kernels. See [vllm-project/vllm#38484](https://github.com/vllm-project/vllm/pull/38484) for the underlying build issue.

---

## Getting Help

If you encounter issues not covered here:

1. Check the [GitHub Issues](https://github.com/razerofficial/aikit/issues) for similar problems
2. Review the [setup documentation](setup.md) for installation requirements
3. Create a new issue with detailed information about your system and the error

## Contributing

Found a solution to a new issue? Please consider:
- Opening a pull request to add it to this document
- Sharing the solution in our community discussions
- Helping other users who encounter similar problems