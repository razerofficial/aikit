# `rzr-aikit`

A command-line interface for Razer AI compute services.

**Usage**:

```console
$ rzr-aikit [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Display help information including usage syntax, available options, and examples.

**Commands**:

* `metrics`: Retrieve model and hardware metrics.
* `model`: Commands to manage models.
* `cluster`: Commands to manage clusters.
* `ui`: Commands to manage the UI server.

## `rzr-aikit metrics`

Retrieve model and hardware metrics.

This command displays comprehensive performance metrics for running models,
including time to first token (TTFT), output throughput for single and parallel
requests, uptime, and GPU resource utilization. Metrics are collected from the
vLLM server&#x27;s Prometheus endpoints.

**Usage**:

```console
$ rzr-aikit metrics [OPTIONS]
```

**Options**:

* `--help`: Display help information including usage syntax, available options, and examples.

## `rzr-aikit model`

Commands to manage models.

**Usage**:

```console
$ rzr-aikit model [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Display help information including usage syntax, available options, and examples.

**Commands**:

* `list`: List all locally cached models.
* `run`: Run a model for inferencing.
* `download`: Download a model from HuggingFace.
* `info`: Get detailed information about a model.
* `remove`: Remove a model from the local cache.
* `stop`: Terminate a running model inference server.
* `generate`: Generate a completion using a model.

### `rzr-aikit model list`

List all locally cached models.

This command displays a table of all models currently stored in the local cache.
For each model, it shows the model name, size on disk, and compatibility status
for both local and distributed inference. Compatibility is determined by comparing
model memory requirements against available GPU memory.

Compatibility indicators:

    ✅ Yes      - Model can run natively with full precision
    🟡 Limited  - Model can run with 4-bit quantization (reduced accuracy)
    ❌ No       - Insufficient memory to run the model
    -           - Distributed compatibility unavailable (not connected to Ray cluster)

**Usage**:

```console
$ rzr-aikit model list [OPTIONS]
```

**Options**:

* `--help`: Show this message and exit.

### `rzr-aikit model run`

Run a model for inferencing.

This command starts a vLLM inference server for the specified model. The server
will automatically optimize parameters based on available GPU resources. You can
pass additional vLLM server arguments after the model name to customize the
configuration. Real-time GPU utilization is displayed during startup (except in
Jupyter environments).

Examples:

    $ rzr-aikit model run deepseek-ai/DeepSeek-R1-Distill-Llama-8B
    $ rzr-aikit model run Qwen/Qwen3-0.6B --max-model-len 4096
    $ rzr-aikit model run facebook/opt-125m --quantization bitsandbytes

**Usage**:

```console
$ rzr-aikit model run [OPTIONS] model [VLLM_SERVE_EXTRA_ARGS...]
```

**Arguments**:

* `model [VLLM_SERVE_EXTRA_ARGS...]`: The name of the model to run, followed by any extra arguments for the VLLM server.  [required]

**Options**:

* `--help`: Show this message and exit.

### `rzr-aikit model download`

Download a model from HuggingFace.

This command downloads the specified model from HuggingFace Hub and stores it
in the local cache directory. The download includes model weights and associated
configuration files. If the model is already cached locally, the command will
skip the download and notify you.

Examples:

    $ rzr-aikit model download Qwen/Qwen3-0.6B
    $ rzr-aikit model download facebook/opt-125m

**Usage**:

```console
$ rzr-aikit model download [OPTIONS] [MODEL_NAME]
```

**Arguments**:

* `[MODEL_NAME]`: organization/model_name  [default: facebook/opt-125m]

**Options**:

* `--help`: Show this message and exit.

### `rzr-aikit model info`

Get detailed information about a model.

This command fetches and displays comprehensive metadata about a model from
HuggingFace Hub, including general information (type, size, downloads, license),
technical specifications (context length, precision, last update), and compatibility
with local and distributed inference. Optionally displays the full model configuration.

Examples:

    $ rzr-aikit model info Qwen/Qwen3-0.6B
    $ rzr-aikit model info deepseek-ai/DeepSeek-R1-Distill-Llama-8B --trust-remote
    $ rzr-aikit model info facebook/opt-125m --full-config

**Usage**:

```console
$ rzr-aikit model info [OPTIONS] [MODEL_NAME]
```

**Arguments**:

* `[MODEL_NAME]`: organization/model_name  [default: facebook/opt-125m]

**Options**:

* `-t, --trust-remote`: Trust remote code
* `-f, --full-config`: Show full model configuration from config.json
* `--help`: Show this message and exit.

### `rzr-aikit model remove`

Remove a model from the local cache.

This command deletes the specified model and all its associated files from the
local HuggingFace cache directory. An error message is displayed if the model is
not found.

Examples:

    $ rzr-aikit model remove microsoft/DialoGPT-small
    $ rzr-aikit model remove facebook/opt-125m

**Usage**:

```console
$ rzr-aikit model remove [OPTIONS] MODEL_NAME
```

**Arguments**:

* `MODEL_NAME`: organization/model_name  [required]

**Options**:

* `--help`: Show this message and exit.

### `rzr-aikit model stop`

Terminate a running model inference server.

This command searches for and stops all running vLLM server processes. It
attempts a graceful shutdown first (SIGTERM), waiting up to 5 seconds. If the
process doesn&#x27;t terminate gracefully, it will force kill (SIGKILL). If multiple
vLLM servers are running, all will be stopped sequentially.

Examples:

    $ rzr-aikit model stop

**Usage**:

```console
$ rzr-aikit model stop [OPTIONS]
```

**Options**:

* `--help`: Show this message and exit.

### `rzr-aikit model generate`

Generate a completion using a model.

This command sends a prompt to the running vLLM server and streams the generated
response to the console. It displays token generation statistics including
prompt tokens, completion tokens, total time, and throughput. If no model is
specified, it uses the first available running model.

Examples:

    $ rzr-aikit model generate &quot;Tell me about AI in Razer&quot;
    $ rzr-aikit model generate &quot;Explain quantum computing&quot; --max-tokens 512
    $ rzr-aikit model generate &quot;Write a poem&quot; --temperature 0.9 --top-p 0.95
    $ rzr-aikit model generate &quot;Hello&quot; --model facebook/opt-125m

**Usage**:

```console
$ rzr-aikit model generate [OPTIONS] PROMPT
```

**Arguments**:

* `PROMPT`: [required]

**Options**:

* `--max-tokens INTEGER`: [default: 256]
* `--temperature FLOAT`: [default: 0.7]
* `--top-p FLOAT`: [default: 1.0]
* `--model TEXT`: Model to use for generation
* `--help`: Show this message and exit.

## `rzr-aikit cluster`

Commands to manage clusters.

**Usage**:

```console
$ rzr-aikit cluster [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Display help information including usage syntax, available options, and examples.

**Commands**:

* `run`: Start cluster head on current node
* `join`: Start a cluster worker on current node to join the cluster
* `stop`: Stop cluster processes on current node
* `status`: Print cluster status

### `rzr-aikit cluster run`

Start a Ray cluster head node.

This command initializes a Ray cluster head node on the current machine with
configurable network interface, dashboard host, CUDA GPU visibility, log level,
and health check timeout. Additional Ray start options can be passed after the
script-specific options.

RAY_START_OPTIONS:

Any additional options will be passed directly to the &#x27;ray start&#x27; command.
These should come after script-specific options or after &#x27;--&#x27;.
For example: --num-cpus 4 --num-gpus 2 --block

Examples:

    $ rzr-aikit cluster run
    $ rzr-aikit cluster run --ifname enp130s0
    $ rzr-aikit cluster run --ifname eth0 --health-check-timeout-ms 60000
    $ rzr-aikit cluster run --dashboard-host 0.0.0.0 --log-level debug

**Usage**:

```console
$ rzr-aikit cluster run [OPTIONS]
```

**Options**:

* `--ifname TEXT`: Specify the network interface for NCCL_SOCKET_IFNAME and GLOO_SOCKET_IFNAME  [default: eth0]
* `--dashboard-host TEXT`: Specify the host IP for the Ray dashboard  [default: 0.0.0.0]
* `--cuda-visible-devices TEXT`: Specify the CUDA_VISIBLE_DEVICES string (e.g., &quot;0&quot;, &quot;0,1&quot;, &quot;1,3&quot;)
* `--log-level TEXT`: Specify the Ray backend log level, None or select one from
* `--health-check-timeout-ms INTEGER`: Specify the Ray health check timeout in milliseconds, None will use Ray&#x27;s internal default 10000 ms
* `--help`: Show this message and exit.

### `rzr-aikit cluster join`

Start a Ray cluster worker node to join an existing cluster.

This command starts a Ray worker node on the current machine and connects it to
an existing Ray cluster head. You must specify the head node address. Additional
configuration options include network interface, CUDA GPU visibility, log level,
and health check timeout. Extra Ray start options can be passed after the
script-specific options.

RAY_START_OPTIONS:

Any additional options will be passed directly to the &#x27;ray start&#x27; command.
These should come after script-specific options or after &#x27;--&#x27;.
For example: --num-cpus 4 --num-gpus 2 --block

Examples:

    $ rzr-aikit cluster join --address 192.168.1.100:6379
    $ rzr-aikit cluster join --ifname enp130s0 --address 192.168.1.100:6379
    $ rzr-aikit cluster join --address 192.168.1.100:6379 --health-check-timeout-ms 60000
    $ rzr-aikit cluster join --address 192.168.1.100:6379 --cuda-visible-devices &quot;0,1&quot;

**Usage**:

```console
$ rzr-aikit cluster join [OPTIONS]
```

**Options**:

* `--ifname TEXT`: Specify the network interface for NCCL_SOCKET_IFNAME and GLOO_SOCKET_IFNAME  [default: eth0]
* `--address TEXT`: Specify Ray head address &lt;Ray head ip&gt;:&lt;Ray head port&gt; to join
* `--cuda-visible-devices TEXT`: Specify the CUDA_VISIBLE_DEVICES string (e.g., &quot;0&quot;, &quot;0,1&quot;, &quot;1,3&quot;)
* `--log-level TEXT`: Specify the Ray backend log level, None or select one from
* `--health-check-timeout-ms INTEGER`: Specify the Ray health check timeout in milliseconds, None will use Ray&#x27;s internal default 10000 ms
* `--help`: Show this message and exit.

### `rzr-aikit cluster stop`

Stop Ray cluster processes on the current node.

This command terminates all Ray processes running on the current machine,
whether it&#x27;s a head node or a worker node. You can optionally force immediate
termination with SIGKILL or specify a grace period for graceful shutdown.
Additional Ray stop options can be passed after the script-specific options.

RAY_STOP_OPTIONS:

Any additional options will be passed directly to the &#x27;ray stop&#x27; command.
For example: --log-style pretty --log-color auto --verbose

Examples:

    $ rzr-aikit cluster stop
    $ rzr-aikit cluster stop --force
    $ rzr-aikit cluster stop --grace-period 30

**Usage**:

```console
$ rzr-aikit cluster stop [OPTIONS]
```

**Options**:

* `--force`: Ray will send SIGKILL instead of SIGTERM
* `--grace-period INTEGER`: The time in seconds ray waits for processes to be properly terminated
* `--help`: Show this message and exit.

### `rzr-aikit cluster status`

Display the current status of the Ray cluster.

This command prints detailed information about the Ray cluster, including node
status, resource utilization, and active workers. If multiple Ray clusters are
running or you need to check a remote cluster, specify the head node address
using the --address option.

Examples:

    $ rzr-aikit cluster status
    $ rzr-aikit cluster status --address 192.168.1.100:6379

**Usage**:

```console
$ rzr-aikit cluster status [OPTIONS]
```

**Options**:

* `--address TEXT`: Override the Ray cluster address to connect to (required for remote clusters or when multiple clusters exist)
* `--help`: Show this message and exit.

## `rzr-aikit ui`

Commands to manage the UI server.

**Usage**:

```console
$ rzr-aikit ui [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Display help information including usage syntax, available options, and examples.

**Commands**:

* `run`: Launch the UI for image generation.
* `stop`: Stop the running UI server.

### `rzr-aikit ui run`

Launch the UI for image generation.

This command starts a local Gradio web UI for image generation and audio
generation, connecting it to a running server endpoint. The Gradio server
runs as a detached background process so the terminal session is not blocked.
Use `rzr-aikit ui stop` to stop it.

Examples:

    $ rzr-aikit ui run
    $ rzr-aikit ui run --server http://127.0.0.1:8000
    $ rzr-aikit ui run --port 7861

**Usage**:

```console
$ rzr-aikit ui run [OPTIONS]
```

**Options**:

* `--server TEXT`: Server URL to connect to for image and audio generation requests  [default: http://localhost:8000]
* `--port INTEGER`: Local port for the Gradio app  [default: 7860]
* `--help`: Show this message and exit.

### `rzr-aikit ui stop`

Stop the running UI server.

This command searches for and stops the background Gradio UI process that
was started with `rzr-aikit ui run`. It attempts a graceful shutdown first
(SIGTERM), waiting up to 5 seconds. If the process does not terminate
gracefully, it will force kill (SIGKILL).

Examples:

    $ rzr-aikit ui stop

**Usage**:

```console
$ rzr-aikit ui stop [OPTIONS]
```

**Options**:

* `--help`: Show this message and exit.