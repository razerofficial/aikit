# Building Razer AIKit

For most use cases, we recommend using the official Razer AIKit container image from Docker Hub for optimal convenience and compatibility.
However, local builds may be necessary for specific scenarios, such as experimental hardware support or custom configurations.
If your use case requires features not covered in the official build, please open an issue and we'll work to address your requirements.

## Building Razer AIKit

Razer AIKit extends the official vLLM image (vllm/vllm-openai) with additional ML tooling, Jupyter integration, and custom GPU capabilities. The build process automatically pulls the latest vLLM base image and adds Razer-specific enhancements.

The container provides comprehensive support for all NVIDIA GPUs with compute capability 7.0 or higher (Workstation/Consumer, Data Center, and Jetson), and has been extensively tested on consumer GPU configurations.

See the [full list of NVIDIA GPUs and their compute capabilities](https://developer.nvidia.com/cuda-gpus).

Build the Razer AIKit container by executing:

```sh
./docker_build_aikit/make.sh
```

Upon successful completion, a local AIKit image will be available with the tag `razer:aikit`.

### Build Configuration

The build uses a multi-stage Dockerfile with the following stages:
1. **Base Environment** - Extends vLLM with Python environment setup
2. **GPU Tools Build** - Compiles custom GPU discovery module
3. **Extended vLLM** - Adds Ray, vLLM audio, scripts, and benchmarks
4. **AIKit Package** - Builds rzr-aikit Python wheel
5. **Test (Optional)** - Runs unit tests
6. **Jupyter Extension** - Builds custom Jupyter Lab extension
7. **Production Image** - Final image with JupyterLab and LlamaFactory

Version pinning is managed in the Dockerfile:
- `CUDA_BUILD_IMAGE` - CUDA development environment (example: nvidia/cuda:12.9.1-devel-ubuntu20.04)
- `VLLM_IMAGE` - vLLM base image (example: vllm/vllm-openai:v0.14.1)

To customize versions or build specific stages, modify the build script in `docker_build_aikit/make.sh`.

## Running Tests

Testing is implemented as a dedicated build target stage in the Razer AIKit Dockerfile. To execute the test suite, run:

```sh
./docker_build_aikit/make.sh test
```

The tests are located in `tests/model/` and include validation of the Razer AIKit CLI functionality.