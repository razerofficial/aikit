# Model Fine-Tuning

Guide to fine-tuning with Razer AIKit using LlamaFactory.

## Overview

Razer AIKit comes with LlamaFactory, which supports multiple fine-tuning algorithms including LoRA, QLoRA, AdaLoRA, DoRA, LongLoRA, LLaMA-Pro, Freeze, and full parameter fine-tuning. In this guide, we'll show a simple example of LoRA fine-tuning on both single devices and distributed clusters.

For advanced fine-tuning techniques and other algorithms, refer to the [LlamaFactory documentation](https://github.com/hiyouga/LLaMA-Factory).

This guide covers:

- LoRA Fine-Tuning Basics - Understanding the LoRA method and its advantages
- Local Fine-Tuning - Training models on a single machine
- Distributed Fine-Tuning - Scaling training across multiple nodes with Ray
- Serving Fine-Tuned Models - Deploying models with LoRA adapters
- Configuration Options - Parameter references
- Practical Examples - Step-by-step workflows

Razer AIKit leverages LlamaFactory for fine-tuning workflows and vLLM for serving fine-tuned models with LoRA adapters.

---

## What is LoRA?

LoRA (Low-Rank Adaptation) is a fine-tuning method that injects small, trainable adapter layers into a model instead of updating all parameters. This approach:

- Reduces memory usage - Only adapter weights are trained (typically <1% of model parameters)
- Enables efficient fine-tuning - Lower compute requirements and faster training
- Prevents catastrophic forgetting - Base model weights remain frozen
- Supports multiple adapters - Switch between different fine-tuned versions dynamically

Trade-offs: LoRA may sacrifice some peak accuracy compared to full fine-tuning, but offers significant efficiency gains for most use cases.

**Learn more:** [Fine-Tuning Large Language Models with LORA: Demystifying Efficient Adaptation](https://medium.com/@kailash.thiyagarajan/fine-tuning-large-language-models-with-lora-demystifying-efficient-adaptation-25fa0a389075)

---

## Serving Fine-Tuned Models

After training, you can serve your base model with LoRA adapters using vLLM. The inference server supports loading multiple LoRA adapters simultaneously and switching between them per request.

### Key Flags for LoRA Serving

When running models with `rzr-aikit model run` or `vllm serve`, use these flags to enable LoRA:

LoRA-Specific Flags:

- `--enable-lora` - Enable LoRA adapter support in the inference server
- `--lora-modules NAME=PATH [NAME=PATH ...]` - Register LoRA adapters with custom names
- `--max-lora-rank INT` - Maximum LoRA rank (default: 16)
- `--max-loras INT` - Maximum number of LoRA adapters to load (default: 1)
- `--max-cpu-loras INT` - Maximum LoRA adapters to cache in CPU memory

Usage Examples:

```bash
# Serve with single LoRA adapter
rzr-aikit model run Qwen/Qwen2.5-0.5B-Instruct \
  --enable-lora \
  --lora-modules alpaca=~/fine-tuning/adapters/lora_alpaca_gpt4

# Serve with multiple LoRA adapters
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --enable-lora \
  --lora-modules \
    alpaca=~/adapters/lora_alpaca \
    medical=~/adapters/lora_medical \
    coding=~/adapters/lora_code \
  --max-loras 3
```

Generating with LoRA Adapters:

```bash
# Use base model (no adapter)
rzr-aikit model generate --model Qwen/Qwen2.5-0.5B-Instruct "Your prompt here"

# Use specific LoRA adapter
rzr-aikit model generate --model alpaca "Your prompt here"
```

---

## Local Fine-Tuning

Local fine-tuning trains LoRA adapters on a single machine using one or more GPUs. This workflow uses LlamaFactory CLI for configuration and training.

### Key Tools

- `llamafactory-cli train` - Execute fine-tuning with specified configuration
- `rzr-aikit model download` - Download base models from HuggingFace
- `rzr-aikit model run` / `vllm serve` - Serve fine-tuned models with LoRA adapters
- `rzr-aikit model generate` - Test model performance

### Dataset Configuration

LlamaFactory uses JSON files to define dataset parameters. Create a `dataset_info.json` file:

```json
{
  "alpaca_gpt4": {
    "hf_hub_url": "vicgalle/alpaca-gpt4",
    "formatting": "alpaca",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  }
}
```

Place this file in a directory (e.g., `~/fine-tuning/data/`) and reference it with `--dataset_dir`.

### Training Configuration

LlamaFactory accepts configuration via command-line flags:

Model & Training Type:

- `--stage sft` - Supervised fine-tuning stage
- `--do_train` - Enable training mode
- `--model_name_or_path TEXT` - Base model path or HuggingFace identifier
- `--template TEXT` - Chat template (e.g., `qwen`, `llama3`, `mistral`)
- `--finetuning_type lora` - Use LoRA fine-tuning

LoRA Parameters:

- `--lora_rank INT` - Low-rank dimension (4-64, typical: 8-16)
- `--lora_alpha INT` - Scaling factor (typically 1-2x rank)
- `--lora_dropout FLOAT` - Dropout rate in LoRA layers (0.0-0.2)
- `--lora_target TEXT` - Target modules (comma-separated)
  - Common: `q_proj,k_proj,v_proj,o_proj` (attention layers)
  - Extended: `q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj`

Dataset & Output:

- `--dataset TEXT` - Dataset name from `dataset_info.json`
- `--dataset_dir PATH` - Directory containing `dataset_info.json`
- `--output_dir PATH` - Output directory for trained adapters
- `--overwrite_output_dir` - Overwrite existing output directory

Training Hyperparameters:

- `--num_train_epochs INT` - Number of training epochs
- `--per_device_train_batch_size INT` - Batch size per GPU
- `--gradient_accumulation_steps INT` - Steps to accumulate gradients
- `--learning_rate FLOAT` - Initial learning rate (typical: 1e-4 to 5e-4)
- `--warmup_ratio FLOAT` - Warmup ratio for learning rate scheduler
- `--cutoff_len INT` - Maximum sequence length
- `--packing BOOL` - Pack multiple samples into sequences
- `--max_samples INT` - Limit training samples (for quick iteration)

Logging & Evaluation:

- `--eval_strategy TEXT` - Evaluation strategy: `no`, `steps`, `epoch`
- `--logging_strategy TEXT` - Logging strategy: `steps`, `epoch`
- `--logging_steps INT` - Log every N steps
- `--save_strategy TEXT` - Checkpoint save strategy: `no`, `steps`, `epoch`
- `--save_steps INT` - Save checkpoint every N steps

Memory & Performance:

- `--gradient_checkpointing` - Trade compute for memory (reduces VRAM usage)
- `--group_by_length` - Group samples by length for efficiency
- `--fp16` - Use 16-bit floating point precision
- `--bf16` - Use bfloat16 precision (if supported)

---

## Distributed Fine-Tuning

Distributed fine-tuning scales training across multiple nodes using Ray backend. This approach parallelizes data processing and gradient computation across cluster GPUs.

### Key Requirements

1. Ray Cluster - Head and worker nodes must be connected
2. Environment Variable - Set `USE_RAY=1` before training
3. YAML Configuration - Ray settings are only available in YAML format
4. Shared Storage - All nodes should have access to dataset and output directories

### Ray Configuration (YAML Only)

Ray-specific settings must be specified in a YAML configuration file:

Ray Training Parameters:

- `ray_run_name: TEXT` - Name for Ray training run
- `ray_num_workers: INT` - Number of Ray workers (typically matches GPU count)
- `ray_storage_path: PATH` - Shared storage path for Ray artifacts
- `placement_strategy: TEXT` - Ray placement strategy (`PACK`, `SPREAD`, `STRICT_PACK`)
- `resources_per_worker:` - Resource allocation per worker
  - `GPU: INT` - Number of GPUs per worker (typically 1)
  - `CPU: INT` - Number of CPUs per worker

Example YAML Configuration:

```yaml
### Model Configuration
model_name_or_path: Qwen/Qwen2.5-0.5B-Instruct
template: qwen

### Training Configuration
stage: sft
do_train: true
finetuning_type: lora

### LoRA Parameters
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.05
lora_target: q_proj,k_proj,v_proj,o_proj

### Dataset and Output
dataset: alpaca_gpt4
dataset_dir: ~/fine-tuning/data
output_dir: ~/fine-tuning/adapters/lora_alpaca_gpt4_distributed
overwrite_output_dir: true

### Training Hyperparameters
num_train_epochs: 1
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 2e-4
warmup_ratio: 0.03
cutoff_len: 768
packing: true
max_samples: 2000

### Logging and Evaluation
eval_strategy: "no"
logging_strategy: steps
logging_steps: 50
save_strategy: "no"

### Memory Optimizations
gradient_checkpointing: true
group_by_length: true
fp16: true

### Ray Distributed Training Configuration
ray_run_name: qwen2p5_lora_distributed
ray_num_workers: 4  # Set to number of GPUs for training
ray_storage_path: ~/fine-tuning/
placement_strategy: PACK
resources_per_worker:
  GPU: 1
```

### Cluster Setup

Use `rzr-aikit cluster` commands to establish Ray cluster (see [inferencing.md](inferencing.md) for detailed cluster setup).

Quick Reference:

```bash
# Head Node
rzr-aikit cluster run --ifname enp47s0

# Worker Node
rzr-aikit cluster join --ifname enp60s0 --address 192.168.8.4:6379

# Check Status
rzr-aikit cluster status
```

---

## Workflow Examples

### Example 1: Local LoRA Fine-Tuning

Workflow for fine-tuning on a single device.

#### 1. Download Base Model

```bash
rzr-aikit model download Qwen/Qwen2.5-0.5B-Instruct
```

#### 2. Create Dataset Configuration

```bash
mkdir -p ~/fine-tuning/data
cat > ~/fine-tuning/data/dataset_info.json << 'JSON'
{
  "alpaca_gpt4": {
    "hf_hub_url": "vicgalle/alpaca-gpt4",
    "formatting": "alpaca",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  }
}
JSON
```

#### 3. Run Fine-Tuning

```bash
llamafactory-cli train \
  --stage sft \
  --do_train \
  --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
  --template qwen \
  --finetuning_type lora \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lora_target q_proj,k_proj,v_proj,o_proj \
  --dataset alpaca_gpt4 \
  --dataset_dir ~/fine-tuning/data \
  --output_dir ~/fine-tuning/adapters/lora_alpaca_gpt4 \
  --overwrite_output_dir \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --warmup_ratio 0.03 \
  --cutoff_len 768 \
  --packing True \
  --max_samples 2000 \
  --eval_strategy no \
  --logging_strategy steps \
  --logging_steps 50 \
  --save_strategy no \
  --gradient_checkpointing \
  --group_by_length \
  --fp16
```

#### 4. Start Inference Server with LoRA Adapter

```bash
nohup vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --enable-lora \
  --lora-modules alpaca=~/fine-tuning/adapters/lora_alpaca_gpt4 \
  --gpu-memory-utilization 0.7 \
  --max-model-len 2048 \
  --enforce-eager \
  > ~/vllm.log 2>&1 &
```

#### 5. Wait for Server Ready

```bash
tail -f ~/vllm.log | sed -n 'p;/Application startup complete\./q'
```

#### 6. Test Base Model

```bash
rzr-aikit model generate --model Qwen/Qwen2.5-0.5B-Instruct "Explain the concept of machine learning in simple terms."
```

#### 7. Test Fine-Tuned Model

```bash
rzr-aikit model generate --model alpaca "Explain the concept of machine learning in simple terms."
```

#### 8. Stop the Server

```bash
rzr-aikit model stop
```

---

### Example 2: Distributed Fine-Tuning

This workflow demonstrates distributed fine-tuning across multiple nodes.

#### Part A: Head Node Setup

#### 1. Download Model on All Nodes

Run on each participating node:

```bash
rzr-aikit model download Qwen/Qwen2.5-0.5B-Instruct
```

#### 2. Identify Network Interface

```bash
ip -f inet -4 -o addr
```

#### 3. Start Cluster Head

```bash
rzr-aikit cluster run --ifname enp47s0
```

#### 4. Verify Cluster Status

```bash
rzr-aikit cluster status
```

Wait for worker nodes to join before proceeding.

#### 5. Create Dataset Configuration

Run on head node (or ensure shared storage):

```bash
mkdir -p ~/fine-tuning/data
cat > ~/fine-tuning/data/dataset_info.json << 'JSON'
{
  "alpaca_gpt4": {
    "hf_hub_url": "vicgalle/alpaca-gpt4",
    "formatting": "alpaca",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  }
}
JSON
```

#### 6. Check GPU Configuration

```bash
gpu-discover --distributed
```

Note the total GPU count across all nodes.

#### 7. Create YAML Configuration

```bash
mkdir -p ~/fine-tuning/configs
cat > ~/fine-tuning/configs/lora_distributed_training.yaml << 'YAML'
### Model Configuration
model_name_or_path: Qwen/Qwen2.5-0.5B-Instruct
template: qwen

### Training Configuration
stage: sft
do_train: true
finetuning_type: lora

### LoRA Parameters
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.05
lora_target: q_proj,k_proj,v_proj,o_proj

### Dataset and Output Configuration
dataset: alpaca_gpt4
dataset_dir: ~/fine-tuning/data
output_dir: ~/fine-tuning/adapters/lora_alpaca_gpt4_distributed
overwrite_output_dir: true

### Training Hyperparameters
num_train_epochs: 1
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 2e-4
warmup_ratio: 0.03
cutoff_len: 768
packing: true
max_samples: 2000

### Logging and Evaluation
eval_strategy: "no"
logging_strategy: steps
logging_steps: 50
save_strategy: "no"

### Memory and Performance Optimizations
gradient_checkpointing: true
group_by_length: true
fp16: true

### Ray Distributed Training Configuration
ray_run_name: qwen2p5_lora_distributed
ray_num_workers: 4  # Set this to the number of GPUs you want to use
ray_storage_path: ~/fine-tuning/
placement_strategy: PACK
resources_per_worker:
  GPU: 1
YAML
```

Note: Adjust `ray_num_workers` based on your cluster GPU count.

#### 8. Run Distributed Fine-Tuning

```bash
export USE_RAY=1
llamafactory-cli train ~/fine-tuning/configs/lora_distributed_training.yaml
```

#### 9. Serve Fine-Tuned Model

```bash
nohup vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --enable-lora \
  --lora-modules alpaca=~/fine-tuning/adapters/lora_alpaca_gpt4_distributed \
  --gpu-memory-utilization 0.7 \
  --max-model-len 2048 \
  --enforce-eager \
  > ~/vllm.log 2>&1 &
```

#### 10. Wait for Server Ready

```bash
tail -f ~/vllm.log | sed -n 'p;/Application startup complete\./q'
```

#### 11. Test Base Model

```bash
rzr-aikit model generate --model Qwen/Qwen2.5-0.5B-Instruct "Explain the concept of machine learning in simple terms."
```

#### 12. Test Fine-Tuned Model

```bash
rzr-aikit model generate --model alpaca "Explain the concept of machine learning in simple terms."
```

#### 13. Stop the Server

```bash
rzr-aikit model stop
```

#### 14. Stop the Cluster

```bash
rzr-aikit cluster stop
```

Run on both head and worker nodes.

---

#### Part B: Worker Node Setup

This part connects worker nodes to the training cluster.

#### 1. Download Model

```bash
rzr-aikit model download Qwen/Qwen2.5-0.5B-Instruct
```

#### 2. Identify Network Interface

```bash
ip -f inet -4 -o addr
```

#### 3. Join Cluster

```bash
rzr-aikit cluster join --ifname enp60s0 --address 192.168.8.4:6379
```
- `--ifname` - Network interface on this worker
- `--address` - Head node IP and cluster port

#### 4. Verify Connection

```bash
rzr-aikit cluster status
```

#### 5. Create Dataset Configuration (Optional)

If not using shared storage, replicate dataset configuration:

```bash
mkdir -p ~/fine-tuning/data
cat > ~/fine-tuning/data/dataset_info.json << 'JSON'
{
  "alpaca_gpt4": {
    "hf_hub_url": "vicgalle/alpaca-gpt4",
    "formatting": "alpaca",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  }
}
JSON
```

#### 7. Stop Worker When Done

```bash
rzr-aikit cluster stop
```

---

## Additional Resources

- CLI Reference - Complete command documentation: [cli-reference.md](cli-reference.md)
- Setup Guide - Installation and environment setup: [setup.md](setup.md)
- Inferencing Guide - Model deployment workflows: [inferencing.md](inferencing.md)
- Interactive Notebooks - Jupyter-based tutorials in `notebooks/` directory:
  - `4_On_Device_Fine_Tuning_LoRA.ipynb` - Local fine-tuning walkthrough
  - `5a_(Head)_Distributed_Fine_Tuning_LoRA.ipynb` - Distributed training (head)
  - `5b_(Node)_Distributed_Fine_Tuning_LoRA.ipynb` - Distributed training (worker)
- LlamaFactory Documentation - LlamaFactory project: https://github.com/hiyouga/LLaMA-Factory
- vLLM LoRA Documentation - LoRA serving with vLLM: https://docs.vllm.ai/en/latest/models/lora.html
- Ray Documentation - Ray distributed computing: https://docs.ray.io