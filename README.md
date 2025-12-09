<div align="center">

# RAZER AIKIT


[![GitHub release (latest by date)](https://img.shields.io/github/v/release/razer/aikit)](https://github.com/razerofficial/aikit/releases)
[![Docker Pulls](https://img.shields.io/docker/pulls/razer/ai_sdk)](https://hub.docker.com/r/razerofficial/aikit)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-available-brightgreen)](docs/)

![Razer AIKIT](docs/images/AIKit_github.png)

[🚀 Quick Start](#quick-start) • [📖 Documentation](docs/) • [🛠️ Contributing](CONTRIBUTING.md)

</div>

---

> **⚠️ Preview Release**
> 
> Razer AIKit is in preview. Features may change or have limitations before stable release. We appreciate your feedback!

---

## What is Razer AIKit?

**Open-source AI development environment** for engineers and researchers. Delivers cloud-grade performance and scalability directly on your desktop with out-of-the-box setup.

### Technical Stack

- **AIKit CLI** - Command-line interface for model lifecycle management
- **vLLM Engine** - Production-grade LLM inference with memory optimization
- **LlamaFactory** - Parameter-efficient fine-tuning framework  
- **Ray** - Distributed computing for seamless multi-GPU scaling

##  Core Features

<table>
<tr>
<td width="50%">

### 🚀 **Local-First AI Development**
Run 280,000+ LLMs locally with full privacy and zero cloud costs.

### 🔧 **Open-Source and Community-Driven**
Apache 2.0 licensed with extensible architecture for custom workflows.

</td>
<td width="50%">

### 🌐 **Intelligent Multi-GPU Scaling**
Ray-based orchestration with automatic resource discovery and load balancing.

### 🎯 **Production-Ready Inference**
vLLM engine with memory optimization, batched inference, and OpenAI-compatible APIs.

</td>
</tr>
</table>

---

##  Quick Start


### Prerequisites

<details>
<summary><b>Windows 11</b></summary>

> **Note**: Razer AIKit runs inside WSL 2 on Windows for optimal performance and compatibility.

- **NVIDIA GPU Driver** - Install [NVIDIA App](https://www.nvidia.com/en-us/software/nvidia-app/) and select **Studio Driver** for best stability
- **WSL 2** - [Microsoft's guide](https://docs.microsoft.com/en-us/windows/wsl/install) 
  - Install 24.04 distribution
  - Configure networking mode to **Mirrored** in WSL Settings
  - **Configure Windows Firewall for WSL**: Run the following command in PowerShell as Administrator to allow incoming connections to WSL:
    ```powershell
    Set-NetFirewallHyperVVMSetting -Name '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' -DefaultInboundAction Allow
    ```
    
- **Docker Engine (WSL)** - [Install guide](https://docs.docker.com/engine/install/ubuntu/)
- **NVIDIA Container Toolkit (WSL)** - [Installation instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

Verify installation:
```bash
nvidia-smi
```

</details>

<details>
<summary><b>Ubuntu 22.04 / 24.04</b></summary>

- **Docker Engine** - [Install guide](https://docs.docker.com/engine/install/ubuntu/)
- **NVIDIA GPU Driver** - Install via `Software & Updates > Additional Drivers`
- **NVIDIA Container Toolkit** - [Installation instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

Verify installation:
```bash
nvidia-smi
```

</details>

### ⚡ One-Line Installation

```bash
# Create huggingface cache directory if it doesn't exist
mkdir -p $HOME/.cache/huggingface

# Pull and run Razer AIKIT
docker run -it \
  --restart=unless-stopped \
  --gpus all \
  --ipc host \
  --network host \
  --mount type=bind,source=$HOME/.cache/huggingface,target=/home/Razer/.cache/huggingface \
  --env HUGGING_FACE_HUB_TOKEN=<YOUR_TOKEN> \
  razerofficial/aikit:latest
```

### 🎯 Your First Model

Once inside the container, choose your preferred approach:

**Option 1: Start with interactive guides**
```bash
# Start Jupyter Lab for interactive guides (guides will be available at the outputted link)
jupyter lab --ip="0.0.0.0"
```
> 💡 **Tip**: Explore the `notebooks/` folder for step-by-step guides and examples!

**OR**

**Option 2: Run a model directly**
```bash
# Run a lightweight coding model
rzr-aikit model run deepseek-ai/deepseek-coder-1.3b-instruct
```



---

## 📚 Documentation & Examples

<table>
<tr>
<td width="50%">

### 📖 **Documentation**
- [CLI Reference](docs/cli-reference.md) - Complete command reference
- [Setup Guide](docs/setup.md) - Detailed installation instructions
- [Fine-Tuning Guide](docs/fine-tuning.md) - Model customization
- [Inference Guide](docs/inferencing.md) - Production deployment
- [Container Guide](docs/build-container.md) - Docker setup
- [Known Issues](docs/known-issues.md) - Common problems and solutions

</td>
<td width="50%">

### 💡 **Interactive Examples**
- [On-Device Inference](notebooks/1_On_Device_Inferencing.ipynb)
- [Distributed Inference](notebooks/2a_(Head)_Distributed_Inferencing.ipynb)
- [Fine-Tuning with LoRA](notebooks/3_On_Device_Fine_Tuning_LoRA.ipynb)
- [OpenAI API Integration](notebooks/5_Integrating_Razer_AIKit_with_OpenAI_API.ipynb)
- [Semantic Search](notebooks/6_Sematic_Search.ipynb)

</td>
</tr>
</table>



### 🏆 **Contributors**

We welcome contributions from the community! Special thanks to all our contributors who make this project possible.

---

## 📄 License & Acknowledgments

**Licensed under [Apache License 2.0](LICENSE)**

**Additional Resources**
- [Security Policy](SECURITY.md) - Security reporting guidelines  
- [Open Source Notices](OPEN_SOURCE_NOTICES.md) - Third-party acknowledgments
- [Contributing Guidelines](CONTRIBUTING.md) - Detailed contribution instructions

---

<div align="center">

**Made with ❤️ by the Razer AI Team**

[🌟 Star us on GitHub](https://github.com/razer/aikit)
</div>
