# Building Razer AIKit

For most use cases, we recommend using the official Razer AIKit container image from Docker Hub for optimal convenience and compatibility.
However, local builds may be necessary for specific scenarios, such as experimental hardware support or custom configurations.
If your use case requires features not covered in the official build, please open an issue and we'll work to address your requirements.

## Building vLLM Container

To build Razer AIKit locally, a vLLM container must first be available as a base dependency.
Razer's custom vLLM build provides comprehensive support for all NVIDIA GPUs (Workstation/Consumer, Data Center, and Jetson) with compute capability 7.0 or higher, and has been extensively tested on consumer GPU configurations. This enhanced compatibility extends beyond the official vLLM image's limited GPU support matrix.

See the [full list of NVIDIA GPUs and their compute capabilities](https://developer.nvidia.com/cuda-gpus).

Build the Razer vLLM container by executing:

```sh
./docker_build_vllm/make.sh
```

Upon successful completion, a local vLLM image will be available with the tag `razer:vllm`.

## Building Razer AIKit

Once the vLLM container is available, build Razer AIKit by running:

```sh
./docker_build_aikit/make.sh
```

**Note**: The build system is currently in active development. Configuration parameters can be modified directly in the Dockerfile and shell script files.
The resulting image will be tagged as `razer:aikit` and can be executed using standard `docker run` commands.

## Running Tests

Testing is implemented as a dedicated build target stage in the Razer AIKit Dockerfile. To execute the test suite, run:

```sh
./docker_build_aikit/make.sh test
```

The tests are located in `rzrtools/rzr/tests/model/` and include validation of the Razer AIKit CLI functionality.