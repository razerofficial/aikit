# GPU Compatibility

This document provides detailed information about NVIDIA GPU compatibility with Razer AIKit.

## Supported GPU Compute Capabilities

Razer AIKit supports NVIDIA GPUs with compute capability 7.0 and higher. The table below shows the supported compute capabilities, architectures, and representative GPU models.

| Compute Capability | Architecture | Representative GPUs |
|-------------------|--------------|---------------------|
| 12.1 | Blackwell | GB10 (DGX Spark) |
| 12.0 | Blackwell | RTX PRO 6000 Blackwell, GeForce RTX 50 series |
| 10.3 | Blackwell | GB300, B300 |
| 10.0 | Blackwell | GB200, B200 |
| 9.0 | Hopper | GH200, H200, H100 |
| 8.9 | Ada Lovelace | L4, L40, L40S, RTX 6000 Ada, RTX 5000 Ada, RTX 4000 Ada, GeForce RTX 40 series |
| 8.6 | Ampere | A40, A10, A16, A2, RTX A6000, RTX A5000, RTX A4000, GeForce RTX 30 series |
| 8.0 | Ampere | A100, A30 |
| 7.5 | Turing | T4, Quadro RTX 8000, Quadro RTX 6000, Quadro RTX 5000, GeForce RTX 20 series |
| 7.0 | Volta | V100, Titan V |

## Checking Your GPU Compute Capability

To check your GPU's compute capability, run:

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

Example output:
```
name, compute_cap
NVIDIA H100 80GB HBM3, 9.0
```