from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# This script is used to build the FlashAttention CUDA custom operator.
# It uses PyTorch's C++ extension utility to compile the CUDA code.

setup(
    name='flash_attn_cuda_lib',  # Name of the Python package that will be created
    ext_modules=[
        CUDAExtension(
            name='flash_attn_cuda_lib',  # Name of the extension module to import in Python
            sources=[
                'flash_attn/cuda/flash_attn_fwd.cu',  # Path to the forward pass CUDA source
                'flash_attn/cuda/flash_attn_bwd.cu',  # Path to the backward pass CUDA source
            ],
            # extra_compile_args={ # Optional: Add compiler flags if needed
            #     'cxx': ['-g'],
            #     'nvcc': ['-O3']
            # }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension  # Command to build the C++/CUDA extension
    }
)
