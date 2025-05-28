from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

ext_modules = [
    CUDAExtension(
        name='flash_attention_cuda',
        sources=[
            'src/flash_attention.cpp',
            'src/flash_attention_fwd.cu',
            'src/flash_attention_bwd.cu',
        ],
        include_dirs=[
            'include',
        ],
        extra_compile_args={
            'cxx': ['-O3', '-std=c++17'],
            'nvcc': [
                '-O3',
                '-std=c++17',
                '--expt-relaxed-constexpr',
                '--extended-lambda',
                '--use_fast_math',
                '-Xptxas=-v',
                '--ptxas-options=-O3',
                '-gencode=arch=compute_80,code=sm_80',  # Ampere
                '-gencode=arch=compute_86,code=sm_86',  # Ampere
                '-gencode=arch=compute_89,code=sm_89',  # Ada Lovelace
                '-gencode=arch=compute_90,code=sm_90',  # Hopper
            ]
        }
    )
]

setup(
    name='flash_attention_cuda',
    version='0.1.0',
    description='Flash Attention CUDA Implementation',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
    python_requires='>=3.8',
    install_requires=[
        'torch>=1.12.0',
        'numpy',
    ]
)
