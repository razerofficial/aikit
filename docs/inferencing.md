# Model Inferencing

Guide to running local and distributed inferencing with Razer AIKit.

## Overview

Razer AIKit uses vLLM for high-performance inference, leveraging PagedAttention memory management, continuous batching, and optimized CUDA kernels for maximum throughput and low latency.

The AIKit CLI automates vLLM configuration by analyzing hardware specifications and model requirements to select optimal parameters. This includes GPU discovery, tensor/pipeline parallelism sizing, context size tuning, and performance optimization.

This guide covers:

- **Local Inferencing** - Running models on a single machine with one or more GPUs
- **Distributed Inferencing** - Scaling models across multiple nodes in a cluster
- **Configuration Options** - Detailed flag references for inference commands
- **Practical Examples** - Step-by-step workflows for common scenarios

---

## Local Inferencing

Local inferencing runs models entirely on a single machine, utilizing one or more GPUs available on that device. This is ideal for models that fit within your local GPU memory constraints.

### Key Commands for Local Inferencing

1. **Get Model Information** - Check model requirements and compatibility
2. **Download Model** - Pull model from HuggingFace to local cache
3. **List Models** - View all locally available models
4. **Run Model** - Start inference server
5. **Generate Text** - Send prompts and receive completions
6. **Benchmark Model** - Measure performance metrics
7. **Stop Model** - Stop the inference server

### Configuration Flags

When running models locally with `rzr-aikit model run`, you can pass additional vLLM configuration flags:

**Common vLLM Flags:**

- `--tensor-parallel-size INT` - Number of GPUs to use for tensor parallelism (default: 1)
- `--pipeline-parallel-size INT` - Number of pipeline stages for pipeline parallelism (default: 1)
- `--max-model-len INT` - Maximum sequence length the model can handle
- `--gpu-memory-utilization FLOAT` - Fraction of GPU memory to use (0.0 to 1.0, default: 0.9)
- `--max-num-seqs INT` - Maximum number of sequences to process in a batch (default: 5)
- `--dtype TEXT` - Data type for model weights: `auto`, `float16`, `bfloat16`, `float32`
- `--enforce-eager` - Disable CUDA graph optimization (useful for debugging)
- `--trust-remote-code` - Allow execution of custom code from model repository
- `--quantization TEXT` - Quantization method: `bitsandbytes`, `gptq`, etc.
- `--served-model-name TEXT` - Model name used in the API

**Performance Tuning:**

- `--disable-log-requests` - Disable request logging for better performance
- `--max-log-len INT` - Maximum length of prompt/output in logs
- `--enable-prefix-caching` - Enable automatic prefix caching

For the complete list of available vLLM configuration options, see the [vLLM Engine Arguments documentation](https://docs.vllm.ai/en/latest/models/engine_args.html).

### GPU Configuration Helpers

Razer AIKit includes helper scripts to optimize GPU configuration:

- `gpu-discover`: Discovers all available GPUs on the current node
- `gpu-select MODEL_NAME`: Returns recommended running command with optimal GPU configuration
- `gpu-filter [OPTIONS]`: Prepares specific GPUs for AI workloads
---

## Distributed Inferencing

Distributed inferencing enables running models that exceed single-machine GPU memory by splitting the workload across multiple nodes in a Ray cluster. This approach uses pipeline parallelism and tensor parallelism to distribute model layers and tensors across GPUs.

### Cluster Configuration

#### Head Node: rzr-aikit cluster run

Starts the Ray cluster head node with the following options:

**Network Configuration:**

- `--ifname TEXT` - Network interface for inter-node communication (default: `eth0`)
  - Examples: `enp47s0`, `enp130s0`, `eno1`
  - Used for NCCL_SOCKET_IFNAME and GLOO_SOCKET_IFNAME
- `--dashboard-host TEXT` - Host IP for Ray dashboard (default: `0.0.0.0`)
  - Set to specific IP to restrict dashboard access
  - Dashboard accessible at `http://<host>:8265`

**Logging & Debugging:**

- `--log-level TEXT` - Ray backend logging verbosity
  - Options: `trace`, `debug`, `info`, `warning`, `error`, `fatal`
  - Default: Ray's internal default (info)

**Health Checks:**

- `--health-check-timeout-ms INT` - Timeout for node health checks in milliseconds
  - Default: 10000 ms (10 seconds)
  - Increase for slower networks or high-latency environments

**Ray-Specific Options:**

Any additional options are passed directly to `ray start`:
- `--num-cpus INT` - Number of CPUs to allocate
- `--num-gpus INT` - Number of GPUs to allocate
- `--block` - Block terminal until cluster is shut down
- `--port INT` - Port for Ray cluster communication (default: 6379)

#### Worker Node: rzr-aikit cluster join

Connects a worker node to the existing cluster head:

**Required Options:**

- `--address TEXT` - **Required** - Ray head address in format `<ip>:<port>`
  - Example: `192.168.8.4:6379`
  - Must match the head node's IP and cluster port

**Network Configuration:**

- `--ifname TEXT` - Network interface for inter-node communication (default: `eth0`)
  - Must be reachable from head node
  - Should match the network segment of the head node

**Logging & Debugging:**

- `--log-level TEXT` - Ray backend logging verbosity
  - Same options as head node

**Health Checks:**

- `--health-check-timeout-ms INT` - Timeout for health checks in milliseconds
  - Should match or exceed head node setting

**Ray-Specific Options:**

Any additional options are passed directly to `ray start`:
- `--num-cpus INT` - Number of CPUs this worker contributes
- `--num-gpus INT` - Number of GPUs this worker contributes
- `--block` - Block terminal until worker is disconnected

### Distributed Model Configuration

After cluster is established, run models with distributed execution:

**Required Flags:**

- `--distributed-executor-backend ray` - Use Ray for distributed execution
- `--tensor-parallel-size INT` - Number of GPUs for tensor parallelism
- `--pipeline-parallel-size INT` - Number of pipeline stages across nodes

**Common Patterns:**

```bash
# Tensor parallelism across 4 GPUs (single or multiple nodes)
rzr-aikit model run MODEL --tensor-parallel-size 4 --distributed-executor-backend ray

# Pipeline parallelism across 2 nodes with 2 GPUs each
rzr-aikit model run MODEL --pipeline-parallel-size 2 --tensor-parallel-size 2 --distributed-executor-backend ray

# Enforce eager mode (disable CUDA graphs, useful for debugging)
rzr-aikit model run MODEL --enforce-eager --distributed-executor-backend ray
```

---

## Workflow Examples

### Example 1: Local On-Device Inferencing

This workflow demonstrates running a small model locally on a single device.

#### 1. Check Model Information

```bash
rzr-aikit model info Qwen/Qwen3-0.6B
```

#### 2. Download the Model

```bash
rzr-aikit model download Qwen/Qwen3-0.6B
```

#### 3. Discover Available GPUs

```bash
gpu-discover
```

#### 4. Get Optimal Configuration (Optional)

```bash
gpu-select Qwen/Qwen3-0.6B
```

#### 5. Run the Model

```bash
rzr-aikit model run Qwen/Qwen3-0.6B
```

Start the inference server with automatic optimization. Or use the recommended command from `gpu-select`:

```bash
# Example with manual configuration
rzr-aikit model run Qwen/Qwen3-0.6B --tensor-parallel-size 1 --gpu-memory-utilization 0.9
```

#### 6. Generate Text

```bash
rzr-aikit model generate "Explain quantum computing to a 12-year-old."
```

#### 7. Benchmark Performance

```bash
python -m vllm.entrypoints.cli.main bench serve \
  --model Qwen/Qwen3-0.6B \
  --dataset-name sharegpt \
  --dataset-path benchmarks/ShareGPT_300_conversations.json \
  --request-rate 10.0 \
  --host 127.0.0.1 \
  --port 8000 \
  --num-prompts 100 \
  --percentile-metrics ttft,tpot,itl,e2el
```

#### 8. Check Metrics

```bash
rzr-aikit metrics
```

#### 9. Stop the Model

```bash
rzr-aikit model stop
```

---

### Example 2: Distributed Inferencing

This workflow sets up distributed model serving across multiple nodes using Ray cluster.

#### Setup Phase - Run on All Nodes

**1. Check Model Requirements**

```bash
rzr-aikit model info Qwen/Qwen3-14B
```

Large models like Qwen3-14B typically require distributed deployment across multiple GPUs or nodes.

**2. Download Model on All Nodes**

Run on each participating node:

```bash
rzr-aikit model download Qwen/Qwen3-14B
```

**3. Identify Network Interface**

Run on each node to find the network interface:

```bash
ip -f inet -4 -o addr
```

Find the network interface that connects to your cluster network (e.g., `enp47s0`, `enp130s0`).

#### Head Node Setup

**4. Start Cluster Head**

```bash
rzr-aikit cluster run --ifname enp47s0
```

Replace `enp47s0` with your actual network interface.

**With additional configuration:**

```bash
rzr-aikit cluster run --ifname enp47s0 --health-check-timeout-ms 60000 --log-level info
```

**5. Verify Cluster Status**

```bash
rzr-aikit cluster status
```

**Access Dashboard:**

Open `http://localhost:8265` in your browser for real-time cluster monitoring.

#### Worker Node Setup

**6. Join Worker Nodes to Cluster**

Run on each worker node:

```bash
rzr-aikit cluster join --ifname enp60s0 --address 192.168.8.4:6379
```

- `--ifname` - Network interface on this worker
- `--address` - Head node IP and cluster port (default port is 6379)

**With additional configuration:**

```bash
rzr-aikit cluster join \
  --ifname enp60s0 \
  --address 192.168.8.4:6379 \
  --health-check-timeout-ms 60000
```

**7. Verify Worker Connection**

Run on head node to confirm workers have joined:

```bash
rzr-aikit cluster status
```

You should see multiple nodes listed.

#### Running the Model - Head Node

**8. Discover Available GPUs**

```bash
gpu-discover
```

**9. Get Optimal Configuration (Optional)**

```bash
gpu-select Qwen/Qwen3-14B
```

Or restrict to specific GPUs:

```bash
gpu-select Qwen/Qwen3-14B --restrict-gpus 0,1,2,3
```

**10. Run Model in Distributed Mode**

```bash
rzr-aikit model run Qwen/Qwen3-14B --enforce-eager --distributed-executor-backend ray
```

Or use the configuration recommended by `gpu-select`:

```bash
# Example with tensor parallelism across 4 GPUs
rzr-aikit model run Qwen/Qwen3-14B \
  --tensor-parallel-size 4 \
  --distributed-executor-backend ray \
  --enforce-eager
```

**11. Generate Text**

```bash
rzr-aikit model generate "Explain quantum computing to a 12-year-old."
```

**12. Benchmark Performance**

```bash
python -m vllm.entrypoints.cli.main bench serve \
  --model Qwen/Qwen3-14B \
  --dataset-name sharegpt \
  --dataset-path benchmarks/ShareGPT_300_conversations.json \
  --request-rate 10.0 \
  --host 127.0.0.1 \
  --port 8000 \
  --num-prompts 10 \
  --percentile-metrics ttft,tpot,itl,e2el
```

#### Cleanup - Run on All Nodes

**13. Stop the Model**

```bash
rzr-aikit model stop
```

**14. Stop the Cluster**

```bash
rzr-aikit cluster stop
```

Run this on both head and worker nodes to fully shut down the cluster.

---

## Additional Resources

- **CLI Reference** - Complete command documentation: [cli-reference.md](cli-reference.md)
- **Setup Guide** - Installation and environment setup: [setup.md](setup.md)
- **Fine-Tuning Guide** - Model training workflows: [fine-tuning.md](fine-tuning.md)
- **Interactive Notebooks** - Jupyter-based tutorials in `notebooks/` directory:
  - `1_On_Device_Inferencing.ipynb` - Local inferencing walkthrough
  - `2a_(Head)_Distributed_Inferencing.ipynb` - Cluster head setup
  - `2b_(Node)_Distributed_Inferencing.ipynb` - Worker node setup
- **vLLM Documentation** - Upstream vLLM project: https://docs.vllm.ai
- **Ray Documentation** - Ray distributed computing: https://docs.ray.io
