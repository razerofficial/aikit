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