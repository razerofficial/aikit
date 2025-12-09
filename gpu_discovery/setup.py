import os
from pathlib import Path
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


# ----------------------------
# Arch flags generator
# ----------------------------
def gen_gencode(arches):
    flags = []
    for a in arches:
        compute = f"compute_{a}"
        sm = f"sm_{a}"
        flags.extend(["-gencode", f"arch={compute},code={sm}"])
    # PTX JIT fallback
    max_arch = arches[-1]
    flags.extend(["-gencode", f"arch=compute_{max_arch},code=compute_{max_arch}"])
    return flags

# Default arch list, overridable with env var
arch_list = os.environ.get("GPU_DISCOVERY_ARCHS", "89,90,100,100a,101,103,120,120a,121")
arches = [x.strip() for x in arch_list.split(",") if x.strip()]

here = Path(__file__).parent.resolve()

include_dirs = [
    str(here),
    str(here / "fp8"),
    str(here / "fp4"),
    str(here / "fp4" / "core"),
    str(here / "fp4" / "cutlass_extensions"),
    str(here / "cutlass" / "include"),
    str(here / "cutlass" / "tools" / "util" / "include"),
]

setup(
    name="gpu_discovery",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="gpu_discovery_cpp",
            sources=[
                "pytorch_binding.cu",
                "fp8/fp8_gemm_verifier.cu",
                "fp4/fp4_sm120a_gemm_verifier.cu",
                "fp4/nvfp4_scaled_mm_sm120_kernels.cu",
            ],
            include_dirs=include_dirs,
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17", "-DNDEBUG"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "-DNDEBUG",
                    "--expt-relaxed-constexpr",
                    "--use_fast_math",
                ] + gen_gencode(arches),
            },
            extra_link_args=[
                    "-O3",
                    "-Wl,--no-as-needed",
                    "-ltorch",
                    "-ltorch_cpu",
                    "-ltorch_cuda",
                    "-lc10",
                    "-lc10_cuda",
                    "-ltorch_python",
                    "-lcudart",
            ]
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    entry_points={
        "console_scripts": [
            "gpu-discovery=gpu_discovery.cli:main",
        ],
    },
    install_requires=["torch>=2.0", "nvidia-ml-py"],
)
