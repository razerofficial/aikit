# setup.py
from setuptools import setup, find_packages

setup(
    name='vllm-plugins',
    version='0.1.1',
    description='Clean vLLM modifications via the plugin system',
    packages=find_packages(),
    install_requires=[
        'vllm>=0.14.0',
        'packaging>=20.0',
    ],
    # Register with vLLM's plugin system
    entry_points={
        'vllm.general_plugins': [
            'RazerRayDistributedExecutor = vllm_plugins:register_RayDistributedExecutor',
            'RazerMultiprocExecutor = vllm_plugins:register_MultiprocExecutor',
            'RazerGPUWorker = vllm_plugins:register_GPUWorker',
            'RazerCoreEngineProcManager= vllm_plugins:register_CoreEngineProcManager',
        ]
    },
    python_requires='>=3.12',
)
