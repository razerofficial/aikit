# Known Issues and Solutions

This document contains common issues you might encounter when using Razer AIKit and their corresponding solutions.

## GPU and Hardware Issues

### 1. NVIDIA Compute Capability Below 8.0

**Problem**: Models fail to run on older NVIDIA GPUs with compute capability below 8.0.

**Solution**: Disable FlashInfer sampler by setting the environment variable before running Razer AIKit:

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
rzr-aikit model run <your-model>
```

**Affected Hardware**: GTX 10 series, RTX 20 series, and some older GPUs.

---

## Memory and Performance Issues

### 2. Windows with VRAM Less Than 10GB

**Problem**: Out of memory errors or poor performance on Windows systems with limited VRAM.

**Solution**: Reduce GPU memory utilization to prevent memory overflow:

```bash
rzr-aikit model run <your-model> --gpu-memory-utilization 0.8
```

**Note**: You can adjust the value (0.8 = 80%) based on your available VRAM. Lower values use less memory but may impact performance.

---

## Platform-Specific Issues

### 3. Windows Inference Not Working

**Problem**: Model inference fails to start or is not accessible on Windows.

**Solution**: Start a local HTTP server to properly expose the inference endpoint:

```bash
python -m http.server --bind 0.0.0.0 8000
```

**Additional Notes**: 
- This is typically needed when running in WSL2 on Windows
- Ensure Windows Firewall allows the connection if needed
- The server will be accessible at `http://localhost:8000`

---

## Getting Help

If you encounter issues not covered here:

1. Check the [GitHub Issues](https://github.com/razer/aikit/issues) for similar problems
2. Review the [setup documentation](setup.md) for installation requirements
3. Create a new issue with detailed information about your system and the error

## Contributing

Found a solution to a new issue? Please consider:
- Opening a pull request to add it to this document
- Sharing the solution in our community discussions
- Helping other users who encounter similar problems