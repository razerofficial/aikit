# Setup Guide

This guide covers installing and configuring Razer AIKit with validation steps for each component.

## System Requirements

### Minimum Requirements
- **OS**: Windows 11 with WSL 2, Ubuntu 22.04/24.04, or compatible Linux distribution
- **RAM**: 8GB (16GB+ recommended)
- **Storage**: 50GB free space (models can be large)
- **Network**: Stable internet connection for initial setup

### Recommended Requirements
- **RAM**: 32GB+ for optimal performance with larger models
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3080, RTX 4070, or better)
- **Storage**: 200GB+ SSD for model caching

## Prerequisites

### Windows 11 Setup

> **Note**: Razer AIKit runs inside WSL 2 on Windows for optimal performance and compatibility.

#### Step 1: Install NVIDIA GPU Driver

1. **Download NVIDIA App**
   - Visit [NVIDIA App](https://www.nvidia.com/en-us/software/nvidia-app/)
   - Download and install the latest version
   - Select **Studio Driver** for best stability

2. **Validate Driver Installation**
   ```powershell
   # Open PowerShell and run:
   nvidia-smi
   ```
   **Expected Output**: GPU information table showing driver version and available memory
   
   **Troubleshooting**: If command not found, restart your system and ensure driver installation completed successfully.

#### Step 2: Install and Configure WSL 2

1. **Install WSL 2**
   ```powershell
   # Open PowerShell as Administrator and run:
   wsl --install
   ```

2. **Install Ubuntu 24.04**
   ```powershell
   # After WSL installation, install Ubuntu 24.04:
   wsl --install -d Ubuntu-24.04
   ```

3. **Configure WSL Settings**
   
   **Option 1: Via WSL Settings App** (Recommended)
   - Open "WSL Settings" from Start menu or search
   - Go to "Networking" settings
   - Set networking mode to **Mirrored**
   
   **Option 2: Via Configuration File**
   - Create or edit `%USERPROFILE%\.wslconfig`:
   ```ini
   [wsl2]
   networkingMode=mirrored
   ```

4. **Validate WSL Installation**
   ```bash
   # Inside WSL, run:
   lsb_release -a
   uname -r
   ```
   **Expected Output**: 
   - Distribution: Ubuntu 24.04
   - Kernel version should include "WSL2"

5. **Configure Windows Firewall for WSL Network Access**
   
   To allow network applications running in WSL to be accessible from Windows and other network devices, configure the Windows Firewall:
   
   ```powershell
   # Open PowerShell as Administrator and run:
   Set-NetFirewallHyperVVMSetting -Name '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' -DefaultInboundAction Allow
   ```
   
   This command allows all incoming connections to go through to WSL. For more information, see [Accessing network applications with WSL | Microsoft Learn](https://learn.microsoft.com/en-us/windows/wsl/networking).

#### Step 3: Install Docker Engine in WSL

1. **Update Package Index**
   ```bash
   sudo apt-get update
   sudo apt-get install -y ca-certificates curl gnupg lsb-release
   ```

2. **Add Docker's Official GPG Key**
   ```bash
   sudo mkdir -p /etc/apt/keyrings
   curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
   ```

3. **Set Up Repository**
   ```bash
   echo \
     "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
     $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
   ```

4. **Install Docker Engine**
   ```bash
   sudo apt-get update
   sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
   ```

5. **Configure Docker for Non-Root User**
   ```bash
   sudo groupadd docker
   sudo usermod -aG docker $USER
   newgrp docker
   ```

6. **Validate Docker Installation**
   ```bash
   docker --version
   docker run hello-world
   ```
   **Expected Output**: 
   - Docker version information
   - "Hello from Docker!" message

#### Step 4: Install NVIDIA Container Toolkit in WSL

1. **Configure Package Repository**
   ```bash
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
       && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
       && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
           sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
           sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   ```

2. **Install Container Toolkit**
   ```bash
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   ```

3. **Configure Docker Runtime**
   ```bash
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```

4. **Validate NVIDIA Container Toolkit**
   ```bash
   nvidia-smi
   docker run --rm --gpus all hello-world
   ```
   **Expected Output**: 
   - First command: GPU information from host
   - Second command: "Hello from Docker!" message (confirms GPU access is configured)

### Ubuntu 22.04 / 24.04 Setup

#### Step 1: Install Docker Engine

1. **Update Package Index**
   ```bash
   sudo apt-get update
   sudo apt-get install -y ca-certificates curl gnupg lsb-release
   ```

2. **Add Docker's Official GPG Key**
   ```bash
   sudo mkdir -p /etc/apt/keyrings
   curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
   ```

3. **Set Up Repository**
   ```bash
   echo \
     "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
     $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
   ```

4. **Install Docker Engine**
   ```bash
   sudo apt-get update
   sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
   ```

5. **Configure Docker for Non-Root User**
   ```bash
   sudo groupadd docker
   sudo usermod -aG docker $USER
   newgrp docker
   ```

6. **Validate Docker Installation**
   ```bash
   docker --version
   docker run hello-world
   sudo systemctl status docker
   ```
   **Expected Output**: 
   - Docker version information
   - "Hello from Docker!" message
   - Docker service status: active (running)

#### Step 2: Install NVIDIA GPU Driver

1. **Install via Software & Updates** (Recommended)
   - Open "Software & Updates" application
   - Go to "Additional Drivers" tab
   - Select the latest NVIDIA driver (usually marked as "tested")
   - Click "Apply Changes" and restart when prompted

2. **Alternative: Command Line Installation**
   ```bash
   # Add NVIDIA PPA
   sudo add-apt-repository ppa:graphics-drivers/ppa
   sudo apt update
   
   # Install latest driver
   sudo ubuntu-drivers autoinstall
   sudo reboot
   ```

3. **Validate Driver Installation**
   ```bash
   nvidia-smi
   ```
   **Expected Output**: 
   - GPU information table

#### Step 3: Install NVIDIA Container Toolkit

1. **Configure Package Repository**
   ```bash
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
       && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
       && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
           sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
           sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   ```

2. **Install Container Toolkit**
   ```bash
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   ```

3. **Configure Docker Runtime**
   ```bash
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```

4. **Validate NVIDIA Container Toolkit**
   ```bash
   docker run --rm --gpus all hello-world
   ```
   **Expected Output**: "Hello from Docker!" message (confirms GPU access is configured)

## Installation

### Step 1: Prepare Environment

1. **Create Hugging Face Cache Directory**
   ```bash
   mkdir -p $HOME/.cache/huggingface
   ```

2. **Validate Directory Creation**
   ```bash
   ls -la $HOME/.cache/
   ```
   **Expected Output**: Directory listing showing `huggingface` folder

3. **Set Up Hugging Face Token** (Optional but Recommended)
   - Visit [Hugging Face Tokens](https://huggingface.co/settings/tokens)
   - Create a new token with these minimum permissions:
     - **Repositories**: "Read access to contents of all public gated repos you can access"
   - For most use cases, you only need repository read access to download gated models
   - Copy the token for later use

### Step 2: Pull and Run Razer AIKit

1. **One-Line Installation**
   ```bash
   # Replace <YOUR_TOKEN> with your actual Hugging Face token
   docker run -it \
     --restart=unless-stopped \
     --gpus all \
     --ipc host \
     --network host \
     --mount type=bind,source=$HOME/.cache/huggingface,target=/home/Razer/.cache/huggingface \
     --env HUGGING_FACE_HUB_TOKEN=<YOUR_TOKEN> \
     razerofficial/aikit:latest
   ```

2. **Validate Container Startup**
   
   **Expected Output**: 
   - Container download progress (first run only)
   - Shell prompt inside container: `Razer@<hostname>:`

3. **Test GPU Access Inside Container**
   ```bash
   # Inside the container, run:
   nvidia-smi
   ```
   **Expected Output**: Same GPU information as host system

### Step 3: Verify Installation

1. **Check Razer AIKit CLI**
   ```bash
   # Inside the container:
   rzr-aikit --help
   ```
   **Expected Output**: Razer AIKit CLI help menu with available commands

2. **Test Basic Functionality**
   ```bash
   # List available models
   rzr-aikit model list
   ```
   **Expected Output**: List of available models

3. **Verify Jupyter Lab** (Optional)
   ```bash
   # Start Jupyter Lab
   jupyter lab --ip="0.0.0.0"
   ```
   **Expected Output**: 
   - Server startup logs
   - Access URL (typically `http://localhost:8888`)

## Your First Model

### Option 1: Interactive Guides (Recommended for Beginners)

1. **Start Jupyter Lab**
   ```bash
   jupyter lab --ip="0.0.0.0"
   ```

2. **Access in Browser**
   - Open the provided URL (e.g., `http://localhost:8888`)
   - Navigate to `notebooks/` folder
   - Start with `1_On_Device_Inferencing.ipynb`

3. **Validate Notebook Access**
   **Expected**: Interactive Jupyter interface with notebook files

### Option 2: Direct Model Execution

1. **Run a Lightweight Model**
   ```bash
   rzr-aikit model run deepseek-ai/deepseek-coder-1.3b-instruct
   ```

2. **Validate Model Loading**
   **Expected Output**: 
   - Model download progress (first run)
   - Model loading logs

3. **Test Model Interaction**
   ```bash
   # Test model generation
   rzr-aikit model generate deepseek-ai/deepseek-coder-1.3b-instruct "Write a simple Python function to add two numbers."
   ```
   **Expected Output**: Generated Python code

## Troubleshooting

For additional troubleshooting guidance, see [Known Issues](known-issues.md).

### Common Issues and Solutions

#### Docker Permission Denied
**Symptom**: `permission denied while trying to connect to the Docker daemon socket`

**Solution**:
```bash
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
# or logout and login again
```

#### NVIDIA GPU Not Detected
**Symptom**: `nvidia-smi` command not found or no GPU information

**Solutions**:
1. **Check driver installation**:
   ```bash
   sudo ubuntu-drivers devices
   sudo ubuntu-drivers autoinstall
   sudo reboot
   ```

2. **Verify container toolkit**:
   ```bash
   sudo systemctl restart docker
   docker run --rm --gpus all hello-world
   ```

#### Container Fails to Start
**Symptom**: Container exits immediately or with error

**Debugging Steps**:
```bash
# Check Docker logs
docker logs <container_id>

# Run without GPU to isolate issue
docker run -it \
  --restart=unless-stopped \
  --ipc host \
  --network host \
  --mount type=bind,source=$HOME/.cache/huggingface,target=/home/Razer/.cache/huggingface \
  razerofficial/aikit:latest bash

# Check system resources
free -h
df -h
```



### Documentation
- **[CLI Reference](cli-reference.md)** - Command reference and examples
- **[Inference Guide](inferencing.md)** - Production deployment and optimization
- **[Fine-Tuning Guide](fine-tuning.md)** - Customize models with your data
- **[Container Guide](build-container.md)** - Build custom containers
- **[Known Issues](known-issues.md)** - Common problems and solutions

### Interactive Examples
- **[On-Device Inference](../notebooks/1_On_Device_Inferencing.ipynb)** - Single GPU inference
- **[Distributed Inference](../notebooks/2a_(Head)_Distributed_Inferencing.ipynb)** - Multi-GPU inference
- **[Fine-Tuning with LoRA](../notebooks/4_On_Device_Fine_Tuning_LoRA.ipynb)** - Parameter-efficient training
- **[OpenAI API Integration](../notebooks/6_Integrating_Razer_AIKit_with_OpenAI_API.ipynb)** - API compatibility
- **[Semantic Search](../notebooks/7_Sematic_Search.ipynb)** - Vector search capabilities

## Getting Help

If you encounter issues not covered in this guide:

1. **Check Known Issues**: Review [known-issues.md](known-issues.md)
2. **Search Documentation**: Full documentation available in `docs/`
3. **Community Support**: Join our community discussions
4. **Bug Reports**: Submit issues with system information

**System Information Template** for bug reports:
```bash
# Run these commands and include output in your report:
uname -a
docker --version
nvidia-smi
cat /etc/os-release
```